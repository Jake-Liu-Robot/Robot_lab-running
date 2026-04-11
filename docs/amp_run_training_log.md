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
| learning_rate | 5e-5 (KLAdaptiveLR) |
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

### 2.5 Checkpoints（保留备查）

```
logs/skrl/g1_amp_run/2026-04-11_13-49-28_amp_torch/checkpoints/
  agent_10000.pt, agent_20000.pt, agent_30000.pt, agent_40000.pt, agent_50000.pt, best_agent.pt
```

---

## 3. 参数调整（2026-04-11，commit 0c5abeb）

| 参数 | Run 1（失败） | Run 2（修改后） | 修改原因 |
|------|-------------|----------------|---------|
| `reset_strategy` | `"random"` | `"default"` | 从站立姿态开始，不是随机跑步帧 |
| `termination_height` | 0.4 | 0.25 | 给更多学习时间，不过早终止 |
| `command_prob_high` | 0.5 | 0.2 | 减少冲刺比例，先学站立和慢走 |
| `command_prob_low` | 0.2 | 0.5 | 增加低速比例，让机器人先学基础 |

**未修改的参数**（保持不变）：
- task_reward_weight = 0.7, style_reward_weight = 0.3
- 速度范围 [0, 4] m/s（未缩小，仍包含所有速度段）
- 网络结构 [1024, 512]
- 其他所有超参数

**设计思路**：
- 这是手动课程学习的第一阶段
- 如果 Run 2 episode_len 增长到 200+ 步，forward_vel 开始跟踪命令 → 调回 prob_high=0.5 继续训练
- 速度范围不变，策略仍然会见到 3-4 m/s 命令（20% 概率），只是频率降低

---

## 4. 训练 Run 2 — 进行中（2026-04-11）

### 4.1 配置

| 参数 | 值 |
|------|-----|
| reset_strategy | "default"（站立姿态） |
| termination_height | 0.25 |
| command_prob_high | 0.2 |
| command_prob_mid | 0.3 |
| command_prob_low | 0.5 |
| 其余 | 同 Run 1 |

**日志目录**：`logs/skrl/g1_amp_run/2026-04-11_15-14-XX_amp_torch/`（待确认）

### 4.2 关键检查点

| 步数 | 检查内容 | 预期 |
|------|---------|------|
| 5K-10K | episode_len 趋势 | 应 > 15 步（比 Run 1 好） |
| 50K | episode_len + forward_vel | episode_len 应持续增长 |
| 100K | forward_vel vs cmd_vel | forward_vel 应开始跟踪 |
| 200K+ | 整体收敛趋势 | 决定是否需要 Run 3 |

### 4.3 监控命令

```bash
# 查看最新 run 的所有指标
/isaac-sim/python.sh -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
dirs = sorted(os.listdir('/workspace/robot_lab/logs/skrl/g1_amp_run/'))
latest = dirs[-1]
ea = EventAccumulator(f'/workspace/robot_lab/logs/skrl/g1_amp_run/{latest}')
ea.Reload()
tags = ea.Tags()['scalars']
for t in tags:
    e = ea.Scalars(t)[-1]
    print(f'{t}: step={e.step}, val={e.value:.4f}')
"

# 查看 episode_len 趋势
/isaac-sim/python.sh -c "
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os
dirs = sorted(os.listdir('/workspace/robot_lab/logs/skrl/g1_amp_run/'))
latest = dirs[-1]
ea = EventAccumulator(f'/workspace/robot_lab/logs/skrl/g1_amp_run/{latest}')
ea.Reload()
[print(f'step={e.step}, episode_len={e.value:.1f}') for e in ea.Scalars('Episode / Total timesteps (mean)')]
"
```

### 4.4 结果

> **（训练进行中，待填充）**

---

## 5. 环境搭建备忘

### 5.1 RunPod 配置

| 项目 | 值 |
|------|-----|
| 模板 | `isaac-lab-2.3.2`（NGC 镜像） |
| GPU | RTX 4090 24GB (~$0.39/hr) |
| 环境变量 | `ACCEPT_EULA=Y` |
| Network Volume | `Isaac sim`，100GB，/workspace |
| Python | `/isaac-sim/python.sh` |

### 5.2 踩坑记录

| # | 问题 | 解决 |
|---|------|------|
| 1 | PyPI `pinocchio` 包 (v0.1) 是假包 | 用 `pip install pin` 安装真正的 Pinocchio (v2.7.0) |
| 2 | `csv2npz_run.py` REPO_ROOT 路径错误 | 6层→7层 `..`（commit 44ff6ae） |
| 3 | URDF mesh 路径 `package://g1_description/...` 找不到 | `--mesh_dir` 指向包含 `g1_description/` 的父目录 |
| 4 | 系统 `pip`/`python3` 被劫持到不存在的路径 | 用 `/isaac-sim/python.sh -m pip` 或 `/isaac-sim/kit/python/bin/python3.11` |
| 5 | TensorBoard 缺 markupsafe/numpy | `/isaac-sim/python.sh -m pip install markupsafe numpy` |
| 6 | RunPod SSH 不支持端口转发 (`-L`) | 用 RunPod HTTP 代理：`https://<pod-id>-6006.proxy.runpod.net` |
| 7 | tmux 中粘贴多行反斜杠命令被拆开 | 用单行命令，不用 `\` 换行 |
| 8 | `reset_strategy="random"` 导致即时摔倒 | 改为 `"default"`（站立姿态重置） |

### 5.3 tmux 使用

```bash
tmux new -s amp                    # 创建 session
Ctrl+B, D                          # 断开
tmux attach -t amp                 # 重连
Ctrl+B, C                          # 新建窗口
Ctrl+B, 0/1/n                      # 切换窗口
tmux ls                            # 列出 sessions

# 需要先安装: apt-get update && apt-get install -y tmux
```

### 5.4 TensorBoard 启动

```bash
# 安装（Isaac Sim python）
/isaac-sim/python.sh -m pip install markupsafe numpy tensorboard

# 启动
/isaac-sim/python.sh -m tensorboard.main --logdir /workspace/robot_lab/logs/skrl/g1_amp_run --port 6006 --bind_all

# 访问方式：RunPod HTTP 代理
# https://<pod-id>-6006.proxy.runpod.net
```

---

## 6. 后续计划

### 手动课程策略

```
Run 2（当前）: default reset, prob_high=0.2, termination=0.25
  ↓ 如果 episode_len > 200 步，forward_vel 跟踪 cmd_vel
Run 3: prob_high=0.5, 可选 reset=random, termination=0.3
  ↓ 用 --checkpoint 接着训练
Run 4: 微调（如有需要）
```

### 收敛判断标准

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| episode_len | 持续增长趋近 1000 | 停滞/下降 → 检查 termination 和 reset |
| disc_accuracy | 55-85% | >95% → 判别器过拟合 |
| forward_vel | 趋近 cmd_vel 均值 | 停滞 → task_reward_weight 太小 |
| rew_velocity | 趋近 1.0+ | <0.1 → 梯度消失 |

### 恢复训练

```bash
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 \
  --algorithm AMP --headless --num_envs 4096 \
  --checkpoint logs/skrl/g1_amp_run/<run_dir>/checkpoints/agent_XXXXX.pt
```
