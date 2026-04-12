# AMP-Run 奖励调参指南

> 基于 G1 跑步任务的实战经验总结
> 适用于 AMP/PPO 类强化学习任务的奖励设计与调参

---

## 1. 奖励设计原则

### 1.1 主次分明

```
主奖励（驱动任务目标）: 权重最大
  例: velocity_tracking = 1.5 × task_weight(0.6) = 0.90
  
辅助奖励（约束行为细节）: 权重较小
  例: upright=1.0, heading=-0.5, yaw_rate=-0.5

规则: 所有辅助惩罚之和 < 主奖励
  否则策略为满足约束放弃主任务
```

### 1.2 惩罚强度的黄金范围

```
惩罚占主奖励的比例:
  < 2%  → 太弱，策略忽略（本项目 base_height=-2.0 只占 1.3%）
  5-20% → 合适，策略注意但不主导
  > 30% → 太强，策略为避免惩罚牺牲主任务（Run 3b 崩溃）

验证方法: eval CSV → 代入公式 → 算实际值 → 对比占比
```

### 1.3 治本优于治标

| 类型 | 例子 | 效果 |
|------|------|------|
| 治标 | `yaw_rate`: 惩罚转动速度 | 不转了但方向已偏，不纠正 |
| **治本** | `heading`: 惩罚偏转角度 | 偏了持续扣分，策略学会纠正 |

### 1.4 条件奖励优于全局奖励

```python
# 全局奖励（所有状态都受影响）
rew_base_height = -3.0 * (h - 0.75)²
# 跑步时 h=0.68 也被惩罚 → 可能影响跑步步态

# 条件奖励（只在特定状态生效）
low_speed_scale = clamp(1.0 - actual_speed, 0, 1)
rew_standing_still = -0.01 * low_speed_scale * joint_vel²
# 跑步时 speed>1 → scale=0 → 完全不影响
```

### 1.5 新增奖励项优于增大现有权重

```
❌ yaw_rate: -0.5 → -1.0（全局影响，Run 3b 崩溃）
✅ 新增 heading: -0.5（精准针对方向偏转，不影响其他）

新增项可以用条件逻辑精准控制作用范围
增大权重会产生全局连锁反应
```

---

## 2. 调参规范

### 2.1 单变量原则

```
每次调参:
  最多改 2-3 个参数
  每个参数变化幅度 ≤ 100%（翻倍）
  
❌ Run 3b: yaw ×2 + base_height ×2.5 → 步态崩溃
✅ Run 4b: base_height +50% + heading +67% + standing ×2 → 安全
```

### 2.2 调参周期

```
修改参数 → 训练 60-80K 步 → eval 评估 → CSV 分析 → 下一轮

60-80K 步的依据:
  < 30K: 策略还没适应新奖励，看不出效果
  60-80K: 策略基本收敛，能判断奖励是否有效
  > 100K: 如果还没效果，奖励信号太弱，再等也没用
```

### 2.3 何时改 vs 何时等

```
┌─────────────────────┐
│ 训练 30-50K 步       │
└──────────┬──────────┘
           ▼
┌─────────────────────┐
│ episode_len 还在涨？ │
├─── 是 ──────────────┤
│ 继续训练，不改参数    │    
└─────────────────────┘    
├─── 否（平台期）──────┤
│          ▼           │
│ ┌───────────────────┐│
│ │ Eval + CSV 分析    ││
│ └────────┬──────────┘│
│          ▼           │
│ ┌───────────────────┐│
│ │ 惩罚占比 < 5%？    ││
│ ├── 是 → 增强奖励    ││
│ ├── 否 → 继续训练    ││
│ └───────────────────┘│
└─────────────────────┘
```

---

## 3. 定量分析方法

### 3.1 用 CSV 诊断奖励强度

```bash
# 1. 跑 eval（关闭域随机化）
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/eval_amp_run.py \
  --checkpoint <path> --cmd_vel ramp --num_envs 2

# 2. 读取 CSV
cat eval_agent_XXXXX_ramp.csv | awk -F',' 'NR%60==0{print}'

# 3. 代入公式计算各项奖励
```

### 3.2 各奖励项的计算公式

```python
# 主奖励
velocity_tracking = 1.5 × exp(-4 × (fwd_vel - cmd_vel)²)
  cmd=4, fwd=3.9: 1.5 × exp(-0.04) = 1.44
  cmd=0, fwd=0.3: 1.5 × exp(-0.36) = 1.05  ← 站着走也能拿高分！

# 辅助奖励
upright = 1.0 × up_z
  站直: 1.0, 前倾15°: 0.97, 倾斜30°: 0.87

base_height = -3.0 × (h - 0.75)²
  h=0.75: 0, h=0.70: -0.008, h=0.65: -0.030

heading = -0.5 × (1 - cos θ)
  0°: 0, 15°: -0.017, 30°: -0.067

standing_still = -0.01 × clamp(1-speed) × Σ(joint_vel²)
  停止+关节动: -0.01 × 1.0 × 500 = -5.0（很强）
  跑步: -0.01 × 0 × anything = 0（关闭）
```

### 3.3 诊断示例（本项目 Run 4）

```
站立阶段 CSV:
  fwd_vel=0.35, h=0.674, cmd=0

各项奖励:
  velocity: 1.5 × exp(-4×0.35²) = 0.91  ← 占总奖励 48%
  upright:  ~0.95                          ← 占 50%
  base_h:   -3.0 × (0.674-0.75)² = -0.017 ← 只占 0.9% ⚠️
  heading:  ~-0.01                          ← 占 0.5%
  
→ base_height 太弱，需要增强
→ velocity 在站立时给分太高（策略没动力完全停下）
```

---

## 4. 常见问题与解决

### 4.1 机器人不摔但步态差

```
原因: episode_len 满分但辅助奖励太弱
解决: CSV 分析 → 找到太弱的惩罚 → 增强
```

### 4.2 调参后步态崩溃

```
原因: 改的参数太多或幅度太大
解决: 回退到上一个好的 checkpoint → 小幅调整
教训: Run 3b（yaw ×2 + base_height ×2.5 → 崩溃）
```

### 4.3 Learning rate 降到 0

```
原因: kl_threshold 太小，KLAdaptiveLR 反复减半
解决: kl_threshold 0.008 → 0.02
```

### 4.4 判别器过拟合（disc_loss < 0.5）

```
原因: 判别器太强，策略得不到 style 梯度
解决: 增大 discriminator_gradient_penalty_scale（5→10）
      或增大 style_reward_weight
```

### 4.5 站立时仍在走（fwd_vel ≠ 0）

```
原因: velocity_tracking 的 exp 衰减太慢
  cmd=0, fwd=0.3 → exp(-0.36) = 0.70 → 仍得 1.05 分
解决: 增大 exp 内的系数（-4 → -8 或 -10）
  或增大 standing_still 惩罚
```

### 4.6 域随机化导致性能下降

```
正常: episode_len 降 15-20%（推扰让部分 env 提前摔倒）
异常: episode_len 降 >50%（推力太大或噪声太强）
解决: 减小 push_force_max 或 obs_noise_std
重要: eval 时必须关闭域随机化
```

---

## 5. 本项目奖励演化历史

```
Run 1:  只有基础奖励 → 即时摔倒
Run 2:  调 reset + termination → 学会站立
Run 2b: 修 kl_threshold → 学会跑步 4m/s ✅
Run 3:  加 upright=1.0 → 上身更直 ✅ 但膝盖弯
Run 3b: yaw=-1.0 + base=-5.0 → ❌ 步态崩溃（教训）
Run 4:  新增 heading + standing_still + 域随机化 → 方向改善
Run 4b: CSV 分析 → 增强弱惩罚 → 进行中
```

### 当前完整奖励配置

```python
# 主奖励
velocity_tracking:  1.5    # × task_weight(0.6) = 0.90

# 姿态
upright:            1.0    # 骨盆直立
base_height:       -3.0    # (h-0.75)² 膝盖高度

# 运动约束
yaw_rate:          -0.5    # 转向速度
lateral_vel:       -0.5    # 侧向速度
heading:           -0.5    # 方向偏转角度
action_rate:       -0.1    # 动作平滑

# 条件奖励
standing_still:    -0.01   # 低速关节安静（speed<1 时生效）

# AMP 风格
task_weight:        0.6
style_weight:       0.4    # 判别器风格约束
```
