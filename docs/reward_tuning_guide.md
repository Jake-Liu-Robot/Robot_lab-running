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

### 1.6 门控陷阱（Run 7 教训）

**条件奖励很有用，但要谨慎设计门控条件**。Run 7 发现 `rew_base_height_run` 被 `run_scale` 门控：

```python
# 有问题的设计
run_scale = clamp(speed - 1.0, 0, 1)
rew_base_height_run = -10 * run_scale * (h - 0.75)²
# speed < 1 时 run_scale=0 → 加速阶段完全不惩罚蹲姿
# → 策略学到"先蹲后跑"，蹲姿成为稳态
```

**诊断信号**：
- pelvis_h 在目标以下但惩罚 logged 值 < -0.01
- 策略在低速阶段（< cfg 阈值）表现异常

**修正**：
```python
# 去掉门控，让惩罚全程生效
rew_base_height_run = -10 * (h - 0.75)²
```

**原则**：核心姿态惩罚（高度、朝向）不应被速度门控 — 任何速度下都应维持正确姿态。门控只适用于真正速度相关的项（如 `rew_standing_still` 专治低速抖动）。

### 1.7 判别器盲点：固定偏差逃过 AMP

**现象**：数据对称但 policy 学到**固定偏差**（如 yaw 固定 8°）

**为什么 AMP 没拦住**：判别器看 3 帧转移（`joint_pos(t), joint_pos(t+1), joint_pos(t+2)`）。静态偏差在所有帧保持一致 → 转移 = 参考转移 → 判别器识别不出。

**识别方法**：对比参考数据的**统计分布**（mean, std, max）和 policy 行为：
```
参考 yaw: mean=0°, std=2°, max=6°  (单帧最大偏差)
Policy:   mean=8° (平均)          ← 超出参考最大值
→ 确认是 policy 学出的 bug
```

**修正**：不依赖判别器，直接强化对应的 env 奖励（如 `rew_heading_run`）。

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

base_height = -10.0 × (h - 0.75)²      # Run 7: 去掉了 run_scale 门控
  h=0.75: 0, h=0.70: -0.025, h=0.65: -0.100, h=0.55: -0.400

heading = -3.0 × run_scale × (1 - cos θ)  # Run 7: -1 → -3 治锁死 yaw
  0°: 0, 8°: -0.029, 15°: -0.102 (run_scale=1)
  # 小角度时梯度 ~sin(θ)，角度越小梯度越弱

standing_still = -0.01 × clamp(1-speed) × Σ(joint_vel²)
  停止+关节动: -0.01 × 1.0 × 500 = -5.0（很强）
  跑步: -0.01 × 0 × anything = 0（关闭）
```

**梯度陷阱**（Run 7 洞察）：
- `(1 - cos θ)` 在小 θ 时梯度消失（~θ/2）→ policy 收敛到 ~8° 后很难继续
- `|θ|`（绝对角度）梯度恒定 → 小 θ 时仍有拉力
- 首选 cos 公式（数值稳定）；若小 θ 卡死，切绝对角度公式但系数需降到 ~1/14

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

### 4.7 策略"蹲姿作弊"（Run 7 教训）

**症状**：
- pelvis_h 明显低于参考（比如 0.54 vs 参考 0.69）
- fwd_vel 能正常跟踪 cmd
- 视觉上是 Groucho walk（深蹲前进）

**诊断**：
```python
# 对比参考数据在相应行为模式下的 pelvis
/isaac-sim/python.sh -c "import numpy as np; d=np.load('<motion.npz>'); bp=d['body_positions']; bv=d['body_linear_velocities']; s=np.linalg.norm(bv[:,0,:2],axis=1); print('Running mean:', bp[s>1.5,0,2].mean())"
# 参考跑步段 0.694  vs  policy 0.54 → 作弊
# 参考跑步段 0.694  vs  policy 0.69 → 正常（不要和全帧 mean 0.72 比）
```

**根因**：
1. `rew_base_height_run` 权重太小（< 速度奖励 5%）
2. `run_scale` 或其他门控导致低速阶段无惩罚
3. `termination_height` 太低（如 0.25）允许深蹲生存

**修复顺序**：
1. `termination_height` 升到接近参考 min（参考 min 0.58 → 设 0.45）
2. 去掉 `rew_base_height_run` 的 `run_scale` 门控
3. `rew_base_height_run` 权重 × 3（-3 → -10）

### 4.8 Yaw 固定偏差（判别器盲点）

**症状**：
- 参考数据 yaw mean=0°（对称）
- policy heading_cos 在 0.99 附近平台化（~8° yaw）
- 数据是镜像增强的

**诊断**：
```python
# 对比参考和 policy 的 yaw 统计
参考 yaw (running): mean=0.00° std=1.98° max=6.39°
policy yaw:         mean=8.1° (超出参考最大值)
→ 确认是 policy 学出的 bug
```

**修复**：
1. 先试 `rew_heading_run ×3`（-1 → -3），保留 cos 公式
2. 20k 步后若仍卡 0.99：
   - 改公式为绝对角度 `rew_heading_run × |yaw|`
   - 但系数需同时降到 ~1/14（避免爆炸）

---

## 5. 本项目奖励演化历史

```
Run 1:  只有基础奖励 → 即时摔倒
Run 2:  调 reset + termination → 学会站立
Run 2b: 修 kl_threshold → 学会跑步 4m/s ✅
Run 3:  加 upright=1.0 → 上身更直 ✅ 但膝盖弯
Run 3b: yaw=-1.0 + base=-5.0 → ❌ 步态崩溃（教训）
Run 4:  新增 heading + standing_still + 域随机化 → 方向改善
Run 4b: CSV 分析 → 增强弱惩罚
Run 5/5b: 多项暴力调参 → ❌ 崩溃
Run 6:  修数据（矫直+镜像+站立）→ 跑步成功但 pelvis=0.54 作弊
Run 7:  base_height×3.3 + 移除 run_scale 门控 + termination 0.45
        → pelvis=0.694 完美匹配参考 ✅
        发现 yaw 8° 锁死 → heading ×3 (进行中)
```

### 当前完整奖励配置（Run 7 Phase 3）

```python
# 主奖励
velocity_tracking:   1.5   # × task_weight(0.5) = 0.75

# 姿态（都不再被 run_scale 门控）
upright:             1.0   # 骨盆直立
rew_base_height_run: -10.0 # (h-0.75)² ×3.3 防作弊
target_base_height:  0.75

# 运动约束（run_scale 门控，speed>1 时生效）
rew_heading_run:     -3.0  # ×3 纠正 8° yaw 锁死
rew_lateral_vel_run: -0.5  # 本体 Y 速度
rew_action_rate:     -0.1  # 动作平滑

# 站立约束（stand_scale 门控，speed<1 时生效）
rew_standing_height: -2.0
rew_standing_still:  -0.01
rew_yaw_rate_stand:  -0.3
rew_heading_stand:   -0.5

# 终止
termination_height:  0.45  # ×1.8 防深蹲存活

# AMP 风格
task_weight:         0.5   # 0.6 → 0.5
style_weight:        0.5   # 0.4 → 0.5
gradient_penalty:    5.0   # 默认值
```
