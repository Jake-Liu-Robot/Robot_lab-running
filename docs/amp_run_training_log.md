# AMP-Run 训练实验日志

> 任务：RobotLab-Isaac-G1-AMP-Run-Direct-v0
> 目标：G1 完整跑步周期 — 站立 → 加速 → 持续高速跑步(~4 m/s) → 减速 → 停止
> 开始日期：2026-04-11

---

## 1. 数据准备（2026-04-11）

### 1.1 数据源

| 项目 | 值 |
|------|-----|
| CSV 文件 | `lafan1_g1/g1/run2_subject1.csv` |
| 帧范围 | [1943-2564]（1-indexed, inclusive） |
| 帧数 | 622 帧 |
| 帧率 | 30 fps |
| 时长 | 20.7s |
| 速度范围 | 2.0 - 3.3 m/s |
| 运动内容 | 站立 → 加速 → 持续跑步 → 减速 → 站立 |

### 1.2 数据转换（csv2npz_run.py）

**转换工具**：`source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz_run.py`
**FK 引擎**：Pinocchio 2.7.0（非 Isaac Sim FK）

**遇到的问题及解决**：

| # | 问题 | 原因 | 解决方案 |
|---|------|------|---------|
| 1 | `FileNotFoundError: .../source/lafan1_g1/...` | `REPO_ROOT` 计算少了一层 `..`（6层→应该7层） | 修复路径（commit 44ff6ae），并用 `--csv` 显式指定 |
| 2 | `AttributeError: module 'pinocchio' has no attribute 'RobotWrapper'` | PyPI 上 `pinocchio` 包（v0.1）是假包，不是机器人学库 | `pip uninstall pinocchio && pip install pin`（真正的 Pinocchio，v2.7.0） |
| 3 | `ValueError: Mesh package://g1_description/meshes/... could not be found` | `--mesh_dir` 指向了 `g1_description/` 本身，而 URDF 用 `package://g1_description/...` | `--mesh_dir` 改为上层目录 `.../unitree/`（包含 `g1_description/`） |
| 4 | URDF 路径（`robot_description/`）不在 RunPod 上 | LAFAN1 数据集的 `robot_description/` 目录未上传 | 用 robot_lab 自带的 URDF：`.../data/Robots/unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf` |

**最终成功命令**：
```bash
/isaac-sim/python.sh source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz_run.py \
  --csv /workspace/robot_lab/lafan1_g1/g1/run2_subject1.csv \
  --urdf /workspace/robot_lab/source/robot_lab/data/Robots/unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf \
  --mesh_dir /workspace/robot_lab/source/robot_lab/data/Robots/unitree
```

**输出 NPZ**：`source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/g1_run2_subject1_30.npz`

```
dof_positions:           (622, 29) float32
dof_velocities:          (622, 29) float32
body_positions:          (622, 14, 3) float32
body_rotations:          (622, 14, 4) float32
body_linear_velocities:  (622, 14, 3) float32
body_angular_velocities: (622, 14, 3) float32
body_names:              14 个 key body
dof_names:               29 个关节
fps:                     30
```

---

## 2. 训练 Run 1 — 失败（2026-04-11）

### 2.1 配置

| 参数 | 值 |
|------|-----|
| 任务 | RobotLab-Isaac-G1-AMP-Run-Direct-v0 |
| 算法 | skrl AMP |
| num_envs | 4096 |
| timesteps | 500,000 |
| rollouts | 16 |
| learning_epochs | 6 |
| mini_batches | 2 |
| learning_rate | 5e-5 (KLAdaptiveLR, kl_threshold=0.008) |
| task_reward_weight | 0.7 |
| style_reward_weight | 0.3 |
| **reset_strategy** | **"random"** |
| **termination_height** | **0.4** |
| **command_prob_high** | **0.5** (50% 冲刺 [3,4] m/s) |
| **command_prob_mid** | **0.3** (30% 慢跑 [1,3] m/s) |
| **command_prob_low** | **0.2** (20% 站立/起步 [0,1] m/s) |

**日志目录**：`logs/skrl/g1_amp_run/2026-04-11_13-49-28_amp_torch/`

### 2.2 训练速度

| 指标 | 值 |
|------|-----|
| 初始速度 | ~16-17 it/s |
| 稳定速度 | ~8-10 it/s（判别器 replay buffer 填充后下降） |
| 预计总时间 | ~10-16 小时 |
| 实际运行 | 到 50,000 步后手动停止 |

### 2.3 结果 — 失败

**50K 步指标**：

| 指标 | 值 | 判断 |
|------|-----|------|
| Episode length (mean) | **15.0 步 ≈ 0.3s** | 严重过短（目标 1000 步 = 20s） |
| Reward (mean) | -2.62 | 负值，惩罚主导 |
| Reward (min) | -1213.3 | 灾难性失败 |
| Discriminator loss | 2.27 | 判别器正常学习 |
| Policy loss | -0.0095 | — |
| Value loss | 0.0148 | — |

**Episode length 趋势（持续下降）**：

```
step=5000,  episode_len=17.3
step=10000, episode_len=16.5
step=15000, episode_len=15.9
step=20000, episode_len=15.6
step=25000, episode_len=15.4
step=30000, episode_len=15.3
step=35000, episode_len=15.2
step=40000, episode_len=15.1
step=45000, episode_len=15.0
step=50000, episode_len=15.0
```

### 2.4 失败分析

**根本原因**：机器人在 0.3 秒内摔倒，且趋势持续恶化。

| 问题 | 说明 |
|------|------|
| `reset_strategy = "random"` | 从参考运动的随机帧重置（跑步中的单脚支撑、腾空等不稳定姿态），机器人还没学会站就被放到跑步姿态 |
| `command_prob_high = 0.5` | 50% 时间命令 3-4 m/s 冲刺，对还不会站的机器人毫无意义 |
| `termination_height = 0.4` | 骨盆低于 0.4m 就终止，机器人刚开始摔就被终止，学不到有用信号 |

**总结**：相当于让一个还不会站的婴儿从跑步姿态开始练冲刺，必然失败。

### 2.5 Checkpoints

```
logs/skrl/g1_amp_run/2026-04-11_13-49-28_amp_torch/checkpoints/
  agent_10000.pt, agent_20000.pt, agent_30000.pt, agent_40000.pt, agent_50000.pt, best_agent.pt
```

---

## 3. 参数调整 #1（2026-04-11，commit 0c5abeb）

| 参数 | Run 1（失败） | Run 2（修改后） | 修改原因 |
|------|-------------|----------------|---------|
| `reset_strategy` | `"random"` | `"default"` | 从站立姿态开始，不是随机跑步帧 |
| `termination_height` | 0.4 | 0.25 | 给更多学习时间，不过早终止 |
| `command_prob_high` | 0.5 | 0.2 | 减少冲刺比例，先学站立和慢走 |
| `command_prob_low` | 0.2 | 0.5 | 增加低速比例，让机器人先学基础 |

**未修改**：task/style 权重、速度范围 [0,4]、网络结构、其他超参数

---

## 4. 训练 Run 2（2026-04-11）

### 4.1 配置

| 参数 | 值 |
|------|-----|
| reset_strategy | "default"（站立姿态） |
| termination_height | 0.25 |
| command_prob_high | 0.2 |
| command_prob_mid | 0.3 |
| command_prob_low | 0.5 |
| kl_threshold | 0.008（未改，导致后续问题） |
| 其余 | 同 Run 1 |

**日志目录**：`logs/skrl/g1_amp_run/2026-04-11_15-14-03_amp_torch/`

### 4.2 Episode Length 趋势 — 大幅改善

```
step=5000,  episode_len=17.9   从站立开始，比 Run 1 同期好
step=10000, episode_len=17.8
step=15000, episode_len=19.5   开始上升
step=20000, episode_len=46.6   突破！学会站稳
step=25000, episode_len=91.0   快速增长
step=30000, episode_len=97.1
step=35000, episode_len=103.7
step=40000, episode_len=108.8
step=45000, episode_len=113.9  增长放缓
step=50000, episode_len=118.2  平台期
```

### 4.3 50K 步指标

| 指标 | Run 2 (50K) | Run 1 (50K) | 改善 |
|------|------------|------------|------|
| Episode length (mean) | **118.2 步 (2.4s)** | 15.0 步 (0.3s) | **7.9x** |
| Episode length (max) | **1199 步 (24s)** | — | 有 env 活过 episode 上限 |
| Reward (mean) | **+0.476** | -2.62 | 转正 |
| Total reward (mean) | **+56.3** | -39.5 | 大幅改善 |
| Discriminator loss | 1.12 | 2.27 | 判别器收敛 |
| Learning rate | **0.0** | — | ⚠️ 被 KLAdaptiveLR 压到 0 |

### 4.4 问题：Learning Rate 降到 0

**现象**：KLAdaptiveLR 将学习率从 5e-5 持续减半至 0.0，策略冻结。

**原因**：`kl_threshold=0.008` 过小。AMP 训练中策略变化大（判别器更新、随机命令、从摔倒到站立的行为剧变），KL 散度频繁超过 2×0.008=0.016，触发 lr 减半。减半速度远快于恢复（翻倍）速度 → lr 单调下降至 0。

**影响**：episode_len 增长从 ~6步/5K 放缓到 ~4步/5K，策略停止改进。

### 4.5 Checkpoints

```
logs/skrl/g1_amp_run/2026-04-11_15-14-03_amp_torch/checkpoints/
  agent_10000.pt ~ agent_50000.pt, best_agent.pt
```

**在 55K 步手动停止。**

---

## 5. 参数调整 #2（2026-04-11，commit 0fb6eec）

| 参数 | Run 2 | Run 2b（续训） | 修改原因 |
|------|-------|---------------|---------|
| `kl_threshold` | 0.008 | **0.02** | 允许更大策略变化，防止 lr 被压到 0 |

---

## 6. 训练 Run 2b — 续训进行中（2026-04-11）

### 6.1 配置

从 Run 2 的 `agent_50000.pt` checkpoint 恢复训练。
唯一修改：`kl_threshold: 0.008 → 0.02`

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 --algorithm AMP \
  --headless --num_envs 4096 \
  --checkpoint logs/skrl/g1_amp_run/2026-04-11_15-14-03_amp_torch/checkpoints/agent_50000.pt
```

**日志目录**：`logs/skrl/g1_amp_run/` 下最新的 `2026-04-11_*` 目录

### 6.2 初始指标（恢复后第一个记录点）

| 指标 | 值 | 说明 |
|------|-----|------|
| Episode length (mean) | 118.5 | 和 Run 2 结束时持平 |
| Learning rate | **4.2e-5** | ✅ 从 0.0 恢复！（初始值 5e-5） |
| Reward (mean) | 0.477 | 持平 |
| Discriminator loss | 1.19 | 正常 |

**LR 修复成功。** 策略恢复更新能力。

### 6.3 Episode Length 趋势（完整）

```
5K   → 118.5    恢复起点
10K  → 132.2    突破平台期
15K  → 154.3
20K  → 253.4    加速增长
25K  → 340.8
30K  → 396.5
35K  → 439.4
40K  → 480.0
45K  → 516.6
50K  → 565.4
55K  → 588.2
60K  → 610.1
65K  → 648.3
70K  → 695.1
75K  → 716.0
80K  → 763.4
85K  → 820.0
90K  → 871.5
95K  → 910.3
100K → 906.3
105K → 956.7
110K → 975.8
115K → 1041.1
120K → 1061.9
125K → 1069.4
130K → 1100.8
135K → 1155.2
140K → 1180.2    接近收敛
145K → 1177.1
150K → 1174.6
155K → 1180.4
160K → 1180.2
165K → 1181.4
170K → 1177.2
175K → 1175.3
180K → 1177.0
185K → 1172.1    收敛在 ~1177 步 (19.6s)
```

### 6.4 185K 步最终指标

| 指标 | 值 |
|------|-----|
| Episode length (mean) | 1172 步 (19.5s) |
| Reward (mean) | 0.55 |
| Total reward (mean) | 644 |
| Learning rate | 1.02e-4 |
| Discriminator loss | 1.75 |

---

## 7. Eval 评估（180K checkpoint）

### 7.1 Ramp 测试：cmd=4.0 (15s) → cmd=0.0 (5s)

**速度跟踪结果（CSV 每秒数据）**：

```
t=1s:   cmd=4.0  fwd=3.40 m/s   加速中
t=2s:   cmd=4.0  fwd=3.93 m/s   接近目标
t=3s:   cmd=4.0  fwd=3.97 m/s   稳定巡航
t=5s:   cmd=4.0  fwd=3.95 m/s
t=8s:   cmd=4.0  fwd=3.98 m/s
t=10s:  cmd=4.0  fwd=3.95 m/s
t=13s:  cmd=4.0  fwd=3.94 m/s
t=15s:  cmd=4.0  fwd=3.94 m/s   最后一秒巡航
t=16s:  cmd=0.0  fwd=0.08 m/s   快速减速！
t=18s:  cmd=0.0  fwd=0.02 m/s   停稳
t=20s:  cmd=0.0  fwd=0.01 m/s   完全静止
```

**结论**：
- ✅ 加速 0→4 m/s：~1.5 秒
- ✅ 巡航 3.9-4.0 m/s：稳定持续 13 秒
- ✅ 减速 4→0 m/s：~1 秒
- ✅ 全程无摔倒，Total reward = 1208

### 7.2 视频观察到的问题

| 问题 | 严重性 | 原因分析 |
|------|--------|---------|
| 跑步方向偏移 | 中 | `rew_yaw_rate=-0.1` 太弱 |
| 站立姿态倾斜+摇晃 | 中 | `rew_upright=0.2` 太弱，`rew_action_rate=-0.05` 太弱 |
| 前脚掌着地 | 无（正常） | 4 m/s 冲刺的自然步态 |
| 第一个 env 偶尔摔倒 | 低 | 初始状态随机性 |

### 7.3 Eval 脚本（eval_amp_run.py）

独立评估脚本，不修改训练配置：
- `--cmd_vel 4.0`：固定速度测试
- `--cmd_vel ramp`：完整周期（4m/s 15s → 0m/s 5s）
- `--cmd_vel random`：训练分布
- 自动录制 20s 视频 + CSV 每步记录
- 摄像头跟踪 robot 0

---

## 8. 参数调整 #3 → Run 3（2026-04-11）

### 8.1 修改内容

基于 180K eval 视频观察的问题：

| 参数 | Run 2b | Run 3 | 目的 |
|------|--------|-------|------|
| `rew_upright` | 0.2 | **1.0** | 修站立倾斜（30°倾斜扣 0.13/步 vs 跑步前倾15°仅扣 0.03/步） |
| `rew_yaw_rate` | -0.1 | **-0.5** | 修跑步方向偏移 |
| `rew_action_rate` | -0.05 | **-0.1** | 修站立摇晃（站立抖动 Σ(Δa²)≈2.0 惩罚 -0.2/步，跑步仅 -0.04/步） |
| `command_prob_high` | 0.2 | **0.4** | 已会跑，增加高速训练（低速从50%降到30%） |

### 8.2 设计权衡分析

**upright=1.0 会不会影响跑步前倾？**
- 跑步前倾 15°：奖励 0.97（vs 满分 1.0，差 0.03/步）
- 速度跟踪奖励：1.5/步 → 策略不会为 0.03 放弃跑步 ✓

**action_rate=-0.1 会不会影响跑步？**
- 跑步周期性步态：连续动作相似，Σ(Δa²)≈0.3-0.5 → 惩罚仅 -0.04/步
- 站立抖动：Σ(Δa²)≈2.0-3.0 → 惩罚 -0.2/步 ← 有效惩罚 ✓

**joint_vel 没有改**：增大 5 倍会严重影响跑步（跑步时关节速度大 → 惩罚可达 -5.0/步）

### 8.3 训练 Run 3 — 进行中

从 180K checkpoint 续训：
```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 --algorithm AMP \
  --headless --num_envs 4096 \
  --checkpoint logs/skrl/g1_amp_run/2026-04-11_16-21-16_amp_torch/checkpoints/agent_180000.pt
```

### 8.4 Run 3 早期指标（20K 步）

```
5K   → 794.7   （奖励变化导致初始下降，正常）
10K  → 1115.9  （快速恢复）
15K  → 1153.6  （接近之前水平）
20K  → 1044.0  （波动，待观察）
```

| 指标 | 值 | 判断 |
|------|-----|------|
| Episode length | 1044 | 波动中，需要更多数据 |
| Reward (mean) | 0.68 | ✅ 高于 Run 2b (0.55)，upright 奖励贡献 |
| Learning rate | 1.3e-4 | ✅ 健康 |
| Discriminator loss | 1.67 | ✅ 健康 |

**待 50K 步后评估趋势。**

---

## 9. 关键经验总结

### 7.1 AMP 训练调参优先级

1. **reset_strategy**：最关键。`"random"` 对复杂运动（跑步）几乎必失败，应从 `"default"` 开始
2. **termination_height**：过高 → episode 太短 → 学不到有效信号
3. **command 分布**：初期偏向低速，让机器人先学站稳
4. **kl_threshold**：AMP 训练策略变化大，默认 0.008 太小，建议 0.02+

### 7.2 判断训练是否正常

| 指标 | 健康信号 | 异常信号 |
|------|---------|---------|
| episode_len | 持续增长 | 下降或平坦 |
| learning_rate | > 0 且稳定 | = 0（策略冻结） |
| reward (mean) | 正值且增长 | 持续负值 |
| discriminator loss | 逐渐下降 | 不下降或爆炸 |

### 7.3 "站立陷阱"分析

**风险**：策略可能卡在站立/慢走，不愿冒摔倒风险去学跑步。

**防护机制**：
- 策略 obs 包含 cmd_vel → 可对不同速度命令输出不同动作
- 判别器奖励跑步风格 → 站着不动 style_reward 低
- 50% 命令要求非零速度 → 站着不动 velocity_reward = 0

**监控**：如果 episode_len > 500 但 forward_vel 停滞在 < 1 m/s → 卡在慢走。
**解决**：提高 command_prob_high 或增大 rew_velocity_tracking。

---

## 8. 环境搭建备忘

### 8.1 RunPod 配置

| 项目 | 值 |
|------|-----|
| 模板 | `isaac-lab-2.3.2`（NGC 镜像） |
| GPU | RTX 4090 24GB (~$0.39/hr) |
| 环境变量 | `ACCEPT_EULA=Y` |
| Network Volume | `Isaac sim`，100GB，/workspace |
| Python | `/isaac-sim/python.sh` |
| Pod ID | x2d2bds0sb4kca |

### 8.2 踩坑记录

| # | 问题 | 解决 |
|---|------|------|
| 1 | PyPI `pinocchio` 包 (v0.1) 是假包 | `pip install pin`（真正的 Pinocchio v2.7.0） |
| 2 | `csv2npz_run.py` REPO_ROOT 路径错误 | 6层→7层 `..`（commit 44ff6ae） |
| 3 | URDF mesh 路径 `package://` 找不到 | `--mesh_dir` 指向包含 `g1_description/` 的父目录 |
| 4 | 系统 `pip`/`python3` 被劫持 | 用 `/isaac-sim/python.sh -m pip` |
| 5 | TensorBoard 缺 markupsafe/numpy | `/isaac-sim/python.sh -m pip install markupsafe numpy tensorboard` |
| 6 | RunPod SSH 不支持 `-L` 端口转发 | 用 HTTP 代理：`https://<pod-id>-6006.proxy.runpod.net` |
| 7 | tmux 中粘贴多行 `\` 命令被拆开 | 用单行命令 |
| 8 | `reset_strategy="random"` 即时摔倒 | 改 `"default"` |
| 9 | `kl_threshold=0.008` → lr=0 | 改 0.02 |
| 10 | 本地 conda 破坏 SSH (`OpenSSL mismatch`) | `export LD_LIBRARY_PATH=""` 后再 ssh |

### 8.3 tmux 使用

```bash
apt-get update && apt-get install -y tmux
tmux new -s amp           # 创建
Ctrl+B, D                 # 断开
tmux attach -t amp        # 重连
Ctrl+B, C                 # 新窗口
Ctrl+B, 0/1/n             # 切换窗口
tmux ls                   # 列出
```

### 8.4 TensorBoard

```bash
/isaac-sim/python.sh -m pip install markupsafe numpy tensorboard
/isaac-sim/python.sh -m tensorboard.main --logdir /workspace/robot_lab/logs/skrl/g1_amp_run --port 6006 --bind_all
# 访问: https://<pod-id>-6006.proxy.runpod.net
```

### 8.5 环境恢复（Pod 重启后）

```bash
source /isaac-sim/setup_python_env.sh
cd /workspace/IsaacLab
/isaac-sim/python.sh -m pip install -e source/isaaclab
/isaac-sim/python.sh -m pip install -e source/isaaclab_assets
/isaac-sim/python.sh -m pip install -e source/isaaclab_tasks
cd /workspace/robot_lab
git pull
/isaac-sim/python.sh -m pip install -e source/robot_lab
/isaac-sim/python.sh -m pip install pin pandas  # 数据转换依赖
apt-get update && apt-get install -y tmux
```

---

## 11. 后续计划

### Run 3 完成后

1. Eval 评估：`--cmd_vel ramp` 对比站立姿态是否改善
2. 如果站立仍然摇晃：考虑增大 `rew_action_rate` → -0.15
3. 如果跑步方向仍偏移：考虑添加全局方向奖励
4. 考虑继续训练到 500K 总步数

### Eval 测试命令

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/eval_amp_run.py \
  --checkpoint <path> --cmd_vel ramp --num_envs 2
```

### 恢复训练命令

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 --algorithm AMP \
  --headless --num_envs 4096 \
  --checkpoint logs/skrl/g1_amp_run/<run_dir>/checkpoints/agent_XXXXX.pt
```

### 监控命令

```bash
# 所有指标
/isaac-sim/python.sh -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import os; d='/workspace/robot_lab/logs/skrl/g1_amp_run/'; latest=sorted(os.listdir(d))[-1]; ea=EventAccumulator(d+latest); ea.Reload(); [print(t.split('/')[-1]+': '+str(round(ea.Scalars(t)[-1].value,6))) for t in ea.Tags()['scalars']]"

# Episode length 趋势
/isaac-sim/python.sh -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import os; d='/workspace/robot_lab/logs/skrl/g1_amp_run/'; latest=sorted(os.listdir(d))[-1]; ea=EventAccumulator(d+latest); ea.Reload(); [print(e.step, round(e.value,1)) for e in ea.Scalars('Episode / Total timesteps (mean)')]"
```

---

## 附录 A：Git Commit 记录

| Commit | 内容 |
|--------|------|
| `44ff6ae` | 修复 csv2npz_run.py: Pinocchio 3.x 兼容 + REPO_ROOT 路径 |
| `0c5abeb` | Run 2: default reset, termination=0.25, prob_high=0.2 |
| `0fb6eec` | 修复 KL threshold: 0.008 → 0.02 |
| `e16acc6` | 修复 play.py: pretrained_checkpoint import |
| `a460e97` | 添加 eval_amp_run.py 评估脚本 |
| `951e389` | Ramp 模式：4m/s 15s → 0m/s 5s |
| `f40aad6` | 地面扩大到 500m × 500m |
| `c698b4f` | Run 3: upright=1.0, yaw_rate=-0.5, prob_high=0.4 |
| `531c48b` | Run 3: action_rate=-0.1 |
| `490f056` | Run 3: upright 确认 1.0 |

## 附录 B：训练历程总览

```
Run 1 (50K步) — 失败
  reset=random, termination=0.4, prob_high=0.5
  episode_len: 17→15（下降，即时摔倒）
  
Run 2 (55K步) — 部分成功
  reset=default, termination=0.25, prob_high=0.2
  episode_len: 17→118（大幅改善，但 lr→0 冻结）

Run 2b (185K步) — 成功
  kl_threshold: 0.008→0.02（修复 lr）
  episode_len: 118→1177（收敛，接近满分）
  Eval: 4.0 m/s 稳定巡航 ✅，站立姿态差 ⚠️

Run 3 (进行中) — 改善姿态
  upright=1.0, yaw_rate=-0.5, action_rate=-0.1, prob_high=0.4
  从 180K checkpoint 续训
  早期: 794→1044（适应新奖励中）
```
