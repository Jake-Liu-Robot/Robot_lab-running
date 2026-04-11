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

### 6.3 监控中 — 待填充

后续检查点：
- 10K 步：episode_len 是否突破 118 平台期
- 50K 步：整体趋势评估
- 100K 步：forward_vel 是否跟踪 cmd_vel

### 6.4 监控命令

```bash
# 查看所有指标最新值
/isaac-sim/python.sh -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import os; d='/workspace/robot_lab/logs/skrl/g1_amp_run/'; latest=sorted(os.listdir(d))[-1]; ea=EventAccumulator(d+latest); ea.Reload(); [print(t.split('/')[-1]+': '+str(round(ea.Scalars(t)[-1].value,6))) for t in ea.Tags()['scalars']]"

# 查看 episode_len 趋势
/isaac-sim/python.sh -c "from tensorboard.backend.event_processing.event_accumulator import EventAccumulator; import os; d='/workspace/robot_lab/logs/skrl/g1_amp_run/'; latest=sorted(os.listdir(d))[-1]; ea=EventAccumulator(d+latest); ea.Reload(); [print(e.step, round(e.value,1)) for e in ea.Scalars('Episode / Total timesteps (mean)')]"
```

---

## 7. 关键经验总结

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

## 9. 后续计划

### 手动课程策略

```
Run 2b（当前）: default reset, prob_high=0.2, kl_threshold=0.02
  ↓ 目标: episode_len > 200+, forward_vel 开始跟踪
Run 3: prob_high=0.5, 可选 termination=0.3
  ↓ --checkpoint 续训，强化高速跑步
Run 4: 微调（如有需要），可选 reset=random
```

### 恢复训练命令

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 --algorithm AMP \
  --headless --num_envs 4096 \
  --checkpoint logs/skrl/g1_amp_run/<run_dir>/checkpoints/agent_XXXXX.pt
```

---

## 附录：Git Commit 记录

| Commit | 内容 |
|--------|------|
| `44ff6ae` | 修复 csv2npz_run.py: Pinocchio 3.x 兼容 + REPO_ROOT 路径 |
| `0c5abeb` | 修复训练: default reset, termination=0.25, prob_high=0.2 |
| `15ad780` | 添加训练日志文档 |
| `0fb6eec` | 修复 KL threshold: 0.008 → 0.02 |
