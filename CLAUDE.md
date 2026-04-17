# CLAUDE.md — robot_lab 双架构 G1 跑步对比实验

## 项目目标

基于 **robot_lab**（https://github.com/fan-ziqi/robot_lab ）使用两种运动模仿架构训练 Unitree G1（29-DOF）人形机器人完成**完整跑步周期**：站立 → 加速 → 长时间高速跑步（~4 m/s）→ 减速 → 停止，并在 MuJoCo 中完成 sim-to-sim 验证：

1. **BeyondMimic**（DeepMimic 增强版）— 锚点相对跟踪奖励，rsl_rl PPO，Manager-Based 工作流
2. **AMP**（对抗运动先验）— 判别器风格奖励 + 速度任务奖励，skrl PPO+AMP，Direct 工作流

**选择 robot_lab**：一个仓库、一套 G1 资产、一个 Isaac Lab 安装、两条训练管线，实验控制变量最少。

**GitHub 同步仓库**: https://github.com/Jake-Liu-Robot/Robot_lab-running.git

---

## 开发环境分工

| 环境 | 用途 | 硬件 |
|------|------|------|
| **本地电脑** | 代码编辑、配置编写、非 FK 数据脚本、MuJoCo sim2sim | 无需 GPU，Python 3.11 |
| **RunPod** | Isaac Lab 训练、csv_to_npz（需 FK）、replay 验证 | RTX 4090 24GB，~$0.39/hr |

⚠️ **本地无法 import 任何 robot_lab 模块**（导入链 `robot_lab → isaaclab → isaacsim → omni.kit` 需要 Isaac Sim）。本地仅编辑代码，不运行环境。

---

## 技术栈

| 组件 | 选型 | 版本 |
|------|------|------|
| 基础项目 | fan-ziqi/robot_lab | **v2.3.0+**（BeyondMimic 最低要求） |
| 仿真 | Isaac Lab + Isaac Sim (PhysX) | 2.3.x / 5.x |
| Python | — | **3.11**（v2.2.0 起从 3.10 升级） |
| BeyondMimic RL | **rsl_rl** (PPO, SGD) | 内置 |
| AMP RL | **skrl** (PPO+AMP, Adam) | 内置 |
| sim2sim | MuJoCo + Menagerie g1_mjx.xml | 3.x |
| 数据 | LAFAN1 + AMASS → G1 29-DOF | CSV → NPZ |

---

## 版本兼容性（已验证）

| robot_lab | Isaac Lab | Isaac Sim | Python | BeyondMimic | AMP Dance |
|-----------|-----------|-----------|--------|:-----------:|:---------:|
| v2.1.1 | 2.1.1 | 4.5.0 | 3.10 | ❌ | ❌ |
| **v2.2.0** | 2.2.0 | 5.0.0 | **3.11** | ❌ | **✅** |
| **v2.3.0** | 2.3.0 | 5.1.0 | 3.11 | **✅** | ✅ |
| **v2.3.2** | 2.3.2 | 4.5/5.0/5.1 | 3.11 | ✅ | ✅ |

⚠️ BeyondMimic 通过 PR #99 合并，首次出现在 v2.3.0。v2.1.1 **不含**该功能。

---

## RunPod 环境搭建

⚠️ **pip 安装 Isaac Sim 有已知问题**（`exts/` 目录缺失导致 `SimulationApp` 为 None）。**必须使用 NGC Docker 镜像。**

### 方式（已验证可用）：NGC Docker 镜像 + Isaac Lab 源码

```bash
# === RunPod Pod 配置 ===
# 镜像: nvcr.io/nvidia/isaac-lab:2.3.2（自定义模板）
# 环境变量: ACCEPT_EULA=Y（必须，否则容器无法启动）
# GPU: RTX 4090, Container Disk: 20GB
# Network Volume: 挂载到 /workspace

# === 进入 Pod 后 ===
source /isaac-sim/setup_python_env.sh

# 安装 Isaac Lab（NGC 镜像有 Isaac Sim 但没有 Isaac Lab）
cd /workspace/IsaacLab
/isaac-sim/python.sh -m pip install -e source/isaaclab
/isaac-sim/python.sh -m pip install -e source/isaaclab_assets
/isaac-sim/python.sh -m pip install -e source/isaaclab_tasks

# 安装 robot_lab
cd /workspace/robot_lab
git pull
/isaac-sim/python.sh -m pip install -e source/robot_lab

# === 验证 ===
/isaac-sim/python.sh scripts/tools/list_envs.py | grep -i "beyondmimic\|amp"
# 应看到:
#   RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0
#   RobotLab-Isaac-G1-AMP-Dance-Direct-v0

# === 关键命令前缀 ===
# 所有 python 命令必须用 /isaac-sim/python.sh 代替 python
# 例如: /isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py ...
```

### RunPod 实际配置（2026-04-09 验证）

- **模板**: `isaac-lab-2.3.2`（自定义，镜像 `nvcr.io/nvidia/isaac-lab:2.3.2`）
- **环境变量**: `ACCEPT_EULA=Y`（必须）
- **GPU**: RTX 4090 24GB（~$0.39/hr）⚠️ 不要用 RTX 5090（Blackwell 有已知 bug）
- **Container Disk**: 20GB
- **Network Volume**: `Isaac sim`，100GB，挂载在 `/workspace`
- **Python**: `/isaac-sim/kit/python/bin/python3.11`（镜像自带，不用 conda）
- **Spot 实例**可用（50-70% 折扣），每 50-100 迭代保存 checkpoint + `--resume`

⚠️ **已踩的坑**：
- pip 安装 isaacsim（5.0.0/5.1.0）`exts/` 目录缺失 → SimulationApp 为 None → 不可用
- Desktop 模板（AI-Dock）无法运行 Isaac Sim，必须用 NGC 镜像
- Conda 环境 Python 版本容易错（3.13 vs 3.11）
- NGC 镜像没有预装 Isaac Lab，需要从 /workspace/IsaacLab 源码安装

### RunPod 环境恢复（Pod 重启后）

```bash
source /isaac-sim/setup_python_env.sh
cd /workspace/robot_lab

# 如果 pip 包丢失（Container Disk 被重置）:
cd /workspace/IsaacLab
/isaac-sim/python.sh -m pip install -e source/isaaclab
/isaac-sim/python.sh -m pip install -e source/isaaclab_assets
/isaac-sim/python.sh -m pip install -e source/isaaclab_tasks
cd /workspace/robot_lab
/isaac-sim/python.sh -m pip install -e source/robot_lab
```

---

## 代码结构（已验证路径）

⚠️ **双层 robot_lab 路径**：外层 `source/robot_lab/` 是 pip 包，内层 `robot_lab/` 是 Python 模块。

```
robot_lab/
├── scripts/
│   ├── reinforcement_learning/
│   │   ├── rsl_rl/train.py          # BeyondMimic 训练
│   │   ├── rsl_rl/play.py           # BeyondMimic 评估
│   │   ├── skrl/train.py            # AMP 训练
│   │   └── skrl/play.py             # AMP 评估
│   └── tools/beyondmimic/
│       ├── csv_to_npz.py            # CSV→NPZ（⚠️ 需 Isaac Sim FK）
│       └── replay_npz.py            # Isaac Sim 回放
│
└── source/robot_lab/robot_lab/      # ← Python 包根
    ├── assets/
    │   ├── unitree.py               # G1 ArticulationCfg（PD增益、关节配置）
    │   └── utils/usd_converter.py   # urdf_to_usd, mjcf_to_usd
    │
    └── tasks/
        ├── manager_based/locomotion/
        │   └── beyondmimic/                     # ← 路线 A
        │       ├── tracking_env_cfg.py          # MDP 完整定义
        │       ├── mdp/
        │       │   ├── commands.py              # 运动加载、自适应采样
        │       │   ├── rewards.py               # DeepMimic 跟踪奖励
        │       │   ├── observations.py          # 非对称 actor-critic 观测
        │       │   ├── terminations.py          # 提前终止
        │       │   └── events.py                # 域随机化
        │       └── config/unitree_g1/
        │           ├── __init__.py              # gym.register()
        │           ├── flat_env_cfg.py           # UnitreeG1BeyondMimicFlatEnvCfg
        │           └── agents/rsl_rl_ppo_cfg.py
        │
        └── direct/g1_amp/                        # ← 路线 B（实际路径）
            ├── g1_amp_env.py                     # 基类（DirectRLEnv + G1AmpEnv）
            ├── g1_amp_env_cfg.py                 # G1AmpDanceEnvCfg
            ├── g1_amp_run_env.py                 # 跑步环境（继承 G1AmpEnv）
            ├── g1_amp_run_env_cfg.py             # 跑步配置（随机速度命令）
            ├── motions/
            │   ├── motion_loader.py              # NPZ 加载器（插值+SLERP）
            │   ├── csv2npz.py                    # CSV→AMP NPZ（Pinocchio FK，舞蹈）
            │   ├── csv2npz_run.py                # CSV→AMP NPZ（Pinocchio FK，跑步）
            │   └── g1_dance1_subject2_30.npz     # 舞蹈参考数据
            ├── agents/
            │   ├── skrl_dance_amp_cfg.yaml       # 舞蹈 AMP 训练配置
            │   └── skrl_run_amp_cfg.yaml         # 跑步 AMP 训练配置
            └── __init__.py                       # gym.register() Dance + Run
```

---

## 运动数据分析（2026-04-09 验证）

### 数据源

| 数据集 | 来源 | 格式 | 位置 |
|--------|------|------|------|
| **LAFAN1 Retargeted** | lvhaidong/LAFAN1_Retargeting_Dataset (HuggingFace) | CSV (36列: pos3+quat4+joints29) | `lafan1_g1/g1/` |
| **AMASS Retargeted** | ember-lab-berkeley/AMASS_Retargeted_for_G1 (HuggingFace) | NPZ (AMP 格式) | `amass_g1_running/g1/` |

### Sprint vs Run 分析

⚠️ **以下速度已修正为 30fps 计算值**（原分析误用 60fps，速度被高估 2 倍）

| | Sprint (sprint1_*) | Run (run1_*, run2_*) |
|--|---------------------|---------------------|
| 运动模式 | 间歇冲刺（冲-停-冲） | 持续跑步 |
| 平均速度 | ~1.0 m/s（大量静止拉低） | 1.2-1.5 m/s |
| 峰值速度 | 4.5-4.9 m/s（短暂） | 3.3-4.4 m/s |
| 最佳10s窗口均速 | ~2.0 m/s | 2.3-2.7 m/s |
| 完整周期(站→跑→站) | 短冲刺+长静止 | 10-20+s，包含自然加减速 |
| **适合目标** | ⚠️ 有4m/s峰值但不持续 | ✅ 持续跑步+自然过渡 |

### 最佳候选（目标：站立→加速→持续跑步→减速→停止）

⚠️ **帧率修正（2026-04-09）**：LAFAN1 实际为 **30fps**（非 60fps），之前所有速度分析被高估 2 倍。

| 排名 | 文件 | 帧范围 | 时长 | 峰值 | 平均速度 | 特点 |
|------|------|--------|------|------|---------|------|
| **1** | **run2_subject1.csv** | **[1943-2564]** | **20.7s** | 3.3 m/s | 2.2 m/s | 完整站→跑→站周期，最稳定持续跑步 |
| 2 | run2_subject4.csv | [3505-3764] | 8.6s | 3.4 m/s | 2.7 m/s | 最快持续段，但无完整起停 |
| 3 | sprint1_subject4.csv | [5950-6250] | 10s | 4.4 m/s | 2.0 m/s | 有4+m/s峰值但冲-停模式 |

### 数据决策

**选择 `run2_subject1.csv` 帧 [1943-2564]（20.7s, 30fps）**：
- 完整周期：静止(0.8s) → 加速 → 持续跑步 2-3.3 m/s(~18s) → 减速 → 静止(0.5s)
- 时长匹配 episode_length=20s
- 实际速度 2-3.3 m/s（非之前错误的 4-6 m/s）
- **两条管线用同一段数据**，通过不同转换工具生成各自格式
- AMP 管线通过速度任务奖励测试能否泛化到 4 m/s

### 数据转换流程

```
run2_subject1.csv [帧 1943-2564, 30fps, 20.7s]
    │
    ├── csv_to_npz.py (Isaac Sim FK, RunPod) → BeyondMimic NPZ
    │     --input_fps 30 --frame_range 1943 2564
    │     keys: joint_pos, joint_vel, body_pos_w, body_quat_w, ...
    │
    └── csv2npz_run.py (Pinocchio FK, RunPod) → AMP NPZ
          --start 1943 --end 2564 --fps 30
          keys: dof_positions, dof_velocities, body_positions, body_rotations, ...
```

---

## 路线 A：BeyondMimic G1 跑步

### 已注册任务

```
ID:    RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0
入口:  isaaclab.envs:ManagerBasedRLEnv
配置:  UnitreeG1BeyondMimicFlatEnvCfg（commit 26e70a3 确认）
RL:    rsl_rl PPO
```

### 奖励（tracking_env_cfg.py RewardsCfg — 不需要修改）

**跟踪奖励**（指数核 `r = exp(-error/σ²)`）：

| 奖励项 | σ | 权重 |
|--------|---|------|
| motion_global_anchor_pos（锚点位置） | 0.3 | 0.5 |
| motion_global_anchor_ori（锚点朝向） | 0.4 | 0.5 |
| motion_body_pos（相对体位置） | 0.3 | 1.0 |
| motion_body_ori（相对体朝向） | 0.4 | 1.0 |
| motion_body_lin_vel（线速度） | 1.0 | 1.0 |
| motion_body_ang_vel（角速度） | 3.14 | 1.0 |

**正则化惩罚**：

| 惩罚项 | 权重 |
|--------|------|
| action_rate_l2 | -0.1 |
| joint_limit | -10.0 |
| undesired_contacts（threshold=1.0） | -0.1 |

### 观测（tracking_env_cfg.py ObservationsCfg）

**Actor（带噪声）**：command, motion_anchor_pos_b(±0.25), motion_anchor_ori_b(±0.05), base_lin_vel(±0.5), base_ang_vel(±0.2), joint_pos(±0.01), joint_vel(±0.5), actions

**Critic（无噪声 + 特权）**：以上全部（无噪声）+ body_pos + body_ori

### 终止条件

| 条件 | 阈值 | 备注 |
|------|------|------|
| time_out | 10.0s | — |
| anchor_pos | 0.25m | Z 轴 |
| anchor_ori | 0.8 | — |
| ee_body_pos | 0.25m | ⚠️ 跑步可能需放宽到 0.35-0.5 |

### 仿真参数

dt=0.005, decimation=4（控制 50Hz）, episode=10s, num_envs=4096, env_spacing=2.5m

### 域随机化

startup: friction[0.3,1.6], joint_default_pos(±0.01rad), torso_COM(±0.025/0.05/0.05)
interval: push_robot([1.0,3.0]s)

### 数据准备（RunPod）

```bash
# CSV → NPZ（需要 Isaac Sim FK）
python scripts/tools/beyondmimic/csv_to_npz.py \
  -f data/sprint1_subject4.csv --input_fps 30 --headless

# NPZ 字段: joint_pos(T,29), joint_vel(T,29), root_pos(T,3),
#           root_quat(T,4)[w,x,y,z], root_lin_vel(T,3), root_ang_vel(T,3), fps

# 回放验证
python scripts/tools/beyondmimic/replay_npz.py -f sprint.npz
```

### 数据转换（RunPod）

```bash
# 舞蹈数据（跑通验证用）
/isaac-sim/python.sh scripts/tools/beyondmimic/csv_to_npz.py \
  -f source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/G1_Take_102.bvh_60hz.csv \
  --input_fps 60 --headless

# 跑步数据（正式实验用）— 先将 run2_subject1.csv 复制到 motion 目录
cp lafan1_g1/g1/run2_subject1.csv \
  source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/
/isaac-sim/python.sh scripts/tools/beyondmimic/csv_to_npz.py \
  -f source/robot_lab/robot_lab/tasks/manager_based/beyondmimic/config/g1/motion/run2_subject1.csv \
  --input_fps 30 --frame_range 1943 2564 --headless
```

### 训练（RunPod）

⚠️ **运动数据路径在 env_cfg 类中指定，不通过 CLI 参数**（`--motion` 不存在）。
⚠️ **所有 python 命令必须用 `/isaac-sim/python.sh` 代替 `python`**。

```bash
# 先在本地修改 flat_env_cfg.py 中的 motion 配置指向跑步 .npz → git push
# RunPod: git pull && /isaac-sim/python.sh -m pip install -e source/robot_lab

/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 \
  --headless --num_envs 4096

# 恢复训练
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 \
  --headless --resume <run_name>

# 评估（带视频录制）
/isaac-sim/python.sh scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --num_envs 2 --video
```

### 需要的修改

| 修改 | 量 | 文件 |
|------|---|------|
| 运动数据路径 | 1 处 | flat_env_cfg.py 或 tracking_env_cfg.py 的 motion 配置 |
| ee_body_pos 阈值 | 可能 1 处 | tracking_env_cfg.py TerminationsCfg（腾空相过早终止时） |
| 奖励/观测/网络 | **0** | 已跨运动类型验证 |

---

## 路线 B：AMP G1 跑步（已实现）

### 已注册任务

```
Dance:  RobotLab-Isaac-G1-AMP-Dance-Direct-v0  (原有，不修改)
Run:    RobotLab-Isaac-G1-AMP-Run-Direct-v0    (新增)
入口:   G1AmpRunEnv (继承 G1AmpEnv, Direct 工作流)
RL:     skrl AMP（仅限 skrl，不可用 rsl_rl）
```

### AMP-Run 核心设计

**设计哲学**：判别器管"风格"（怎么跑），速度命令管"任务"（跑多快），两者解耦。

**三网络架构**：Policy [1024,512] + Value [1024,512] + Discriminator [1024,512]

**观测分离**（关键：防止判别器阻止速度泛化）：
```
Policy obs (109维): AMP obs(105) + root_vel_body(3) + cmd_vel(1)
  → 策略知道当前速度和目标速度
AMP obs (105维):    joint_pos(29) + joint_vel(29) + height(1) + orient(6) + key_body(39) + progress(1)
  → 判别器看不到速度命令，不会惩罚"跑得比参考快"
```

**奖励组合**（skrl 内部，Phase 4 起）：
```
r_total = 0.5 × r_task (env返回) + 0.5 × r_style (判别器)
# Phase 4 将 0.7/0.3 → 0.5/0.5：判别器被过度弱化时无法纠正步态（跑步 gait 回归站立抬腿）
# grad_penalty 保持 5.0（Phase 4 短暂降到 3.0 后回退，见 commit 99ec0bb）
```

**速度命令**（随机偏向高速采样）：
```
每 3-7s 采样一个新命令（偏向高速）：
  50% → [3, 4] m/s（冲刺练习）
  30% → [1, 3] m/s（慢跑）
  20% → [0, 1] m/s（站立/起步）
→ 策略自然学会加速、巡航、减速、站立
→ 部署时发送任意速度序列（如 0→4→4→...→0）
→ 不绑定 episode 时长，4 m/s 巡航可持续任意长度
→ 切换均匀采样：command_prob_high=0.25, command_prob_mid=0.5
```

**环境 task_reward** (`g1_amp_run_env.py`，Phase 5 生效中):
```
# 共享（所有速度）
velocity_tracking:  1.5 × exp(-4·(v_wx - cmd)²)            ← 世界 X 方向速度跟踪
upright:            1.0 × pelvis_up_z                       ← 保持直立
rew_base_height_run: -10.0 × (h - 0.75)²                    ← 重心高度约束（Phase 4: -3→-10, 移除 run_scale 门控）
action_rate:       -0.1 × Σ(Δa²)                            ← 动作平滑

# 跑步段（cmd_vel > 1，run_scale = clamp(cmd-1, 0, 1)）
rew_heading_run:   -10.0 × run_scale × (1 - cos(yaw))       ← Phase 5: -5→-10 破 7° yaw plateau
rew_lateral_vel_run: -0.5 × run_scale × vy²                 ← 抑制侧移

# 站立段（cmd_vel < 1，stand_scale = clamp(1-cmd, 0, 1)）
rew_standing_height: -2.0 × stand_scale × (h - 0.75)²
rew_standing_still:  -0.01 × stand_scale × |joint_vel|
rew_yaw_rate_stand:  -0.3 × stand_scale × ωz²
rew_heading_stand:   -0.5 × stand_scale × (1 - cos(yaw))

# 终止
termination_height: 0.45   ← Phase 4: 0.25→0.45, 防深蹲作弊（当前 h=0.698，25cm safety margin）

注意：无显式模仿奖励——判别器已承担风格约束
```

**判别器 style_reward**:
```
输入: amp_obs = 3帧 × 105维 = 315维 (关节角+速度+body位置+朝向+进度)
正样本: 参考运动数据 (学习"跑步风格")
负样本: 策略产生的运动
r_style = 2.0 × max(1 - 0.25·(1 - D_logit)², 0.0001)
gradient_penalty = 5.0 (防止判别器过拟合)
```

**motion_speed（可选数据加速）**：
```
默认 1.0（原速）。如果判别器阻止高速跑步（forward_vel 停滞 + disc_accuracy > 95%），
可设 1.3-1.5 加速参考数据（3.3 m/s → 4.3-5.0 m/s），代价是低速段 style reward 降低。
```

### 已实现的文件

| 文件 | 作用 |
|------|------|
| `g1_amp/__init__.py` | 注册 Dance + Run 两个任务 |
| `g1_amp/g1_amp_run_env.py` | 继承 G1AmpEnv，随机速度命令 + 纯任务奖励 |
| `g1_amp/g1_amp_run_env_cfg.py` | 配置：cmd_vel[0,4], episode=20s, obs=109, termination_height=0.45 |
| `g1_amp/agents/skrl_run_amp_cfg.yaml` | task_w=0.5, style_w=0.5（Phase 4） |
| `g1_amp/motions/csv2npz_run.py` | 参数化数据转换脚本（Pinocchio FK） |
| `docs/amp_run_training_log.md` | **完整训练调参日志** — Run 1→7 + Phase 3/4/5，含每次改动的原因和结果 |

### 训练（RunPod）

```bash
# AMP 数据转换
/isaac-sim/python.sh -m pip install pinocchio
/isaac-sim/python.sh source/robot_lab/robot_lab/tasks/direct/g1_amp/motions/csv2npz_run.py

# 训练
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 \
  --algorithm AMP --headless --num_envs 4096

# 评估
/isaac-sim/python.sh scripts/reinforcement_learning/skrl/play.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 \
  --algorithm AMP --num_envs 32
```

### AMP 训练监控

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| disc_accuracy | 55-85% | >95% → 增大 gradient_penalty 或设 motion_speed=1.3 |
| forward_vel | 趋近 cmd_vel 均值(~2) | 停滞 → 增大 task_reward_weight |
| cmd_vel vs forward_vel | 两者趋势一致 | 不跟踪 → 检查 obs 是否包含 cmd_vel |
| episode_length | 趋近 20s（1000 步 = 满格 24s） | <2s → termination_height 过高 |
| rew_velocity | 趋近 1.0+ | 持续 <0.1 → velocity reward 梯度消失，靠判别器 bootstrap |
| **pelvis_height（作弊警报）** | **≥ 0.72（参考 mean），Phase 5 当前 0.698** | **≥3 cm 下跌 → 策略发现新作弊路径，立即停训排查（见 Phase 4 squat-cheat 事件）** |
| **heading_cos** | **> 0.99（Phase 5 目标）** | **卡在 0.97（~7° yaw plateau）→ 提高 heading 权重或切 abs 公式** |

⚠️ **恢复训练只用 `agent_<iter>.pt`，不要用 `best_agent.pt`**：best 按 eval return 选，可能滞后真实训练状态；`agent_<iter>.pt` 精确对应某一步。

⚠️ **TB 已启用 per-term 奖励日志**（`Reward/rew_velocity`, `Reward/rew_base_height_run`, `Reward/rew_heading_run` 等）→ 应监控每项而非仅 total_reward。

---

## BeyondMimic vs AMP 架构对比

### 与 MimicKit 的 5 大架构差异

| 维度 | MimicKit DeepMimic | robot_lab BeyondMimic | MimicKit AMP | robot_lab AMP |
|------|-------------------|----------------------|--------------|--------------|
| 跟踪方式 | 全局坐标 | **锚点相对**（漂移容忍） | — | — |
| RSI 采样 | 均匀 | **自适应**（偏向困难段） | — | — |
| Actor-Critic | 对称 | **非对称**（critic 有特权信息） | 对称 | 对称 |
| RL 库 | 内置 PPO | **rsl_rl** (外部) | 内置 PPO+AMP | **skrl** (外部) |
| 运动格式 | .pkl（指数映射） | **.npz**（FK 最大坐标） | .pkl | **.npz** |

### 两条路线的实验控制

| 变量 | BeyondMimic | AMP | 对齐 |
|------|-------------|-----|:----:|
| G1 资产 | unitree.py | unitree.py | ✅ |
| 并行环境 | 4096 | 4096 | ✅ |
| 运动数据 | sprint .npz | 同源 sprint | ✅ |
| 训练预算 | N steps | N steps | ✅ |
| 随机种子 | 5 seeds | 5 seeds | ✅ |
| RL 库 | rsl_rl (SGD) | skrl (Adam) | ⚠️ 不可控 |
| 工作流 | Manager-Based | Direct | ⚠️ 不可控 |

---

## MuJoCo Sim-to-Sim（本地）

### G1 模型

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git
# 使用: unitree_g1/g1_mjx.xml
```

### 对齐要点

| 项目 | Isaac Lab | MuJoCo | 转换 |
|------|-----------|--------|------|
| 关节顺序 | URDF | MJCF | 建立显式映射表 |
| 四元数 | [x,y,z,w] | [w,x,y,z] | 重排列 |
| 基座速度 | body frame | world frame | R^T @ vel |
| PD 增益 | unitree.py | general actuator | gainprm/biasprm |

### 策略导出

- BeyondMimic (rsl_rl): 自动导出 ONNX
- AMP (skrl): `torch.onnx.export` 或 `torch.jit.save`

### sim2sim 参考实现

- Mini-Pi-Plus_BeyondMimic: `scripts/sim2sim.py`（最接近）
- unitree_rl_lab: `deploy/robots/g1_29dof/`（官方）
- Humanoid-Gym: `sim2sim.py`（通用）

---

## 本地↔RunPod 工作流

```
本地电脑（无 GPU）
  ├─ 编辑 env_cfg, rewards, observations
  ├─ 创建 AMP-Run 任务代码
  ├─ 编写数据转换脚本
  └─ git push
        │
        ▼
RunPod（RTX 4090）
  ├─ git pull && pip install -e source/robot_lab
  ├─ csv_to_npz.py（需 Isaac Sim FK）
  ├─ train.py --headless（BeyondMimic / AMP）
  ├─ play.py 评估
  └─ 导出 ONNX → scp 或 Network Volume
        │
        ▼
本地电脑
  ├─ 下载 ONNX checkpoint
  ├─ MuJoCo sim-to-sim 验证
  └─ 结果分析、可视化、对比报告
```

---

## 开发时间线

| 阶段 | 地点 | 时间 | 任务 |
|------|------|------|------|
| P1 环境验证 | RunPod | 1 天 | 安装 robot_lab，验证两个任务注册，用默认数据各跑 256 env 小测 |
| P2 数据准备 | RunPod+本地 | 1 天 | LAFAN1 sprint CSV→NPZ，AMP 数据准备，回放验证 |
| P3 BeyondMimic | RunPod | 2-3 天 | 5 seeds × 4096 env × ~2-4hr/seed |
| P4 AMP-Run | 本地+RunPod | 2-3 天 | 本地创建任务→RunPod 训练 5 seeds |
| P5 验证+对比 | 本地 | 2-3 天 | MuJoCo sim2sim，收集指标，对比报告 |

---

## 收集指标

```python
# 跟踪精度
joint_pos_rmse, root_pos_error, end_effector_error

# 运动质量
forward_velocity (vs 4.0), froude_number (>1=跑步),
flight_phase_ratio, step_frequency

# 效率
cost_of_transport = Energy/(m·g·d)

# 鲁棒性
survival_time, perturbation_survival (100/200/300N)

# 训练效率
convergence_iterations, wall_clock_time
```

---

## 降级方案

| 风险 | 降级 |
|------|------|
| v2.3.0 安装失败 | Docker `nvcr.io/nvidia/isaac-lab:2.3.2` |
| BeyondMimic 任务缺失 | 用 main 分支而非 tag |
| AMP 判别器坍缩 | 降 gradient_penalty 到 1.0，增 replay buffer |
| 跑步过早终止 | 放宽 ee_body_pos 到 0.35-0.5 |
| 4m/s 数据不可用 | 用已有 g1_run.pkl 2.8m/s + time-warp |
| sim2sim 差距大 | 检查关节映射/四元数/PD 增益 |
| 时间不足 | 仅完成 BeyondMimic，跳过 AMP |

---

## 关键仓库

| 仓库 | 用途 |
|------|------|
| fan-ziqi/robot_lab | 主项目 |
| HybridRobotics/whole_body_tracking | BeyondMimic 原版参考 |
| linden713/humanoid_amp | AMP G1 上游 |
| lvhaidong/LAFAN1_Retargeting_Dataset | LAFAN1 数据 |
| ember-lab-berkeley/AMASS_Retargeted_for_G1 | AMASS 数据 |
| google-deepmind/mujoco_menagerie | G1 MuJoCo 模型 |
| HighTorque-Robotics/Mini-Pi-Plus_BeyondMimic | sim2sim 参考 |
| unitreerobotics/unitree_rl_lab | 官方 G1 RL+部署 |
| fan-ziqi/rl_sar | 部署框架 |

---

## 开发约定

- 所有新代码本地编辑，git 同步到 RunPod
- 不修改 robot_lab 核心代码，只添加新任务配置
- 训练日志 TensorBoard（`--logger tb`）或 WandB（`--logger wandb`）
- 每个实验 5 个随机种子
- Spot 实例每 50-100 迭代保存 checkpoint
- 最终策略导出 ONNX
