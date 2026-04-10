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

```bash
# === 方式 A：NGC Docker（推荐） ===
# RunPod 选择镜像: nvcr.io/nvidia/isaac-lab:2.3.2
# 容器已有 Isaac Lab，只装 robot_lab：

git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab && git checkout v2.3.2
pip install -e source/robot_lab

# === 方式 B：pip 从零安装 ===
conda create -n isaaclab python=3.11 -y && conda activate isaaclab
pip install isaacsim==5.1.0 isaacsim-extscache-physics==5.1.0 \
    isaacsim-extscache-kit==5.1.0 isaacsim-extscache-kit-sdk==5.1.0 \
    --extra-index-url https://pypi.nvidia.com
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab && git checkout v2.3.0 && ./isaaclab.sh --install && cd ..
git clone https://github.com/fan-ziqi/robot_lab.git
cd robot_lab && git checkout v2.3.2 && pip install -e source/robot_lab

# === 验证（两种方式都执行） ===
python scripts/tools/list_envs.py | grep -i "beyondmimic\|amp"
# 应看到:
#   RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0
#   RobotLab-Isaac-G1-AMP-Dance-Direct-v0
```

### RunPod 实际配置（2026-04-09 创建）

- **模板**: Linux Desktop - AI-Dock（ghcr.io/ai-dock/linux-desktop:latest）— 带桌面环境用于可视化
- **GPU**: RTX 4090 24GB（~$0.39/hr）⚠️ 不要用 RTX 5090（Blackwell 有已知 bug）
- **Container Disk**: 50GB
- **Network Volume**: `Isaac sim`，100GB，挂载在 `/workspace`
- **CUDA**: 12.8（nvidia-smi 确认）
- **Conda**: miniconda3 安装在 `/workspace/miniconda3`（持久化）
- **Conda 环境**: `isaaclab`，Python 3.11.15 ⚠️ 必须用 3.11，**不要用 3.13**
- **Spot 实例**可用（50-70% 折扣），每 50-100 迭代保存 checkpoint + `--resume`

### RunPod 环境恢复（Pod 重启后）

```bash
export PATH="/workspace/miniconda3/bin:$PATH"
conda activate isaaclab
cd /workspace/robot_lab
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
            ├── motions/
            │   ├── motion_loader.py              # NPZ 加载器
            │   ├── csv2npz.py                    # CSV→AMP NPZ（Pinocchio FK）
            │   └── g1_dance1_subject2_30.npz     # 舞蹈参考数据
            ├── agents/skrl_dance_amp_cfg.yaml
            └── __init__.py                       # gym.register()
```

---

## 运动数据分析（2026-04-09 验证）

### 数据源

| 数据集 | 来源 | 格式 | 位置 |
|--------|------|------|------|
| **LAFAN1 Retargeted** | lvhaidong/LAFAN1_Retargeting_Dataset (HuggingFace) | CSV (36列: pos3+quat4+joints29) | `lafan1_g1/g1/` |
| **AMASS Retargeted** | ember-lab-berkeley/AMASS_Retargeted_for_G1 (HuggingFace) | NPZ (AMP 格式) | `amass_g1_running/g1/` |

### Sprint vs Run 分析

| | Sprint (sprint1_*) | Run (run1_*, run2_*) |
|--|---------------------|---------------------|
| 运动模式 | 间歇冲刺（冲-停-冲） | 持续跑步 |
| 平均速度 | ~1.9 m/s（大量静止拉低） | 2.4-3.1 m/s |
| 峰值速度 | 6-9 m/s（远超目标） | 5-7 m/s |
| 单次跑步持续 | 2-3s | 5-50+s |
| 完整周期(站→跑→站) | 3-6s，跑步段太短 | 5-50+s，包含自然加减速 |
| **适合目标** | ❌ 冲刺太短太快 | ✅ 持续跑步+自然过渡 |

### 最佳候选（目标：站立→加速→持续~4m/s→减速→停止）

| 排名 | 文件 | 帧范围 | 时长 | 峰值 | 持续跑(>3m/s) | 特点 |
|------|------|--------|------|------|--------------|------|
| **1** | **run2_subject1.csv** | [1967-2532] | **9.4s** | 6.6 m/s | 8.6s | 完美周期：加速→4-5m/s稳态→减速，时长适中 |
| 2 | run2_subject1.csv | [150-968] | 13.6s | 5.4 m/s | 12.0s | 较长但速度偏低(3-4m/s) |
| 3 | run2_subject4.csv | [78-2340] | 37.7s | 6.8 m/s | 31.4s | 很长但中间有速度波动 |
| 4 | run2_subject1.csv | [5711-7316] | 26.8s | 7.5 m/s | 22.6s | 长但峰值高 |

### 数据决策

**选择 `run2_subject1.csv` 帧 [1967-2532]（9.4s）**：
- 完整周期：静止(0.7s) → 加速(0.3s) → 持续跑步 4-6 m/s(8.6s) → 减速停止(0.4s)
- 时长匹配 AMP episode_length=10s
- 速度集中在 4-5 m/s，贴近目标
- **两条管线用同一段数据**，通过不同转换工具生成各自格式

### 数据转换流程

```
run2_subject1.csv [帧 1967-2532, 60fps]
    │
    ├── csv_to_npz.py (Isaac Sim FK, RunPod) → BeyondMimic NPZ
    │     keys: joint_pos, joint_vel, body_pos_w, body_quat_w, ...
    │
    └── csv2npz.py (Pinocchio FK, RunPod/本地) → AMP NPZ
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

### 训练（RunPod）

⚠️ **运动数据路径在 env_cfg 类中指定，不通过 CLI 参数**（`--motion` 不存在）。

```bash
# 先在本地修改 flat_env_cfg.py 中的 motion 配置指向跑步 .npz → git push
# RunPod: git pull && pip install -e source/robot_lab

python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 \
  --headless --num_envs 4096

# 恢复训练
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 \
  --headless --resume <run_name>

# 评估
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=RobotLab-Isaac-BeyondMimic-Flat-Unitree-G1-v0 --num_envs 2
```

### 需要的修改

| 修改 | 量 | 文件 |
|------|---|------|
| 运动数据路径 | 1 处 | flat_env_cfg.py 或 tracking_env_cfg.py 的 motion 配置 |
| ee_body_pos 阈值 | 可能 1 处 | tracking_env_cfg.py TerminationsCfg（腾空相过早终止时） |
| 奖励/观测/网络 | **0** | 已跨运动类型验证 |

---

## 路线 B：AMP G1 跑步

### 已注册任务（当前为 Dance）

```
ID:    RobotLab-Isaac-G1-AMP-Dance-Direct-v0
入口:  robot_lab.tasks.direct.amp...:G1AmpEnv（Direct 工作流）
RL:    skrl AMP（仅限 skrl，不可用 rsl_rl）
```

### AMP 判别器默认配置

| 参数 | 值 |
|------|---|
| 隐藏层 | [1024, 512] ReLU |
| discriminator_reward_scale | 2.0 |
| discriminator_gradient_penalty_scale | 5.0 |
| task_reward_weight | **0.0**（Dance 纯风格） |
| style_reward_weight | **1.0** |
| num_amp_observations | 2（当前 + 1 历史帧） |

风格奖励：`r = max(1 - 0.25·(1 - D_logit)², 0.0001) × 2.0`

### 创建 AMP-Run 任务（本地编辑，RunPod 训练）

#### Step 1：复制 Dance 配置（本地）

```bash
cd source/robot_lab/robot_lab/tasks/direct/amp/config/
cp -r unitree_g1 unitree_g1_run
```

#### Step 2：添加速度跟踪奖励（本地编辑）

在 `unitree_g1_run/g1_amp_run_env_cfg.py` 中：

```python
import torch

# 修改 1: 奖励权重
task_reward_weight = 0.5    # 从 0.0 → 0.5
style_reward_weight = 0.5   # 从 1.0 → 0.5

# 修改 2: 添加速度跟踪任务奖励
def _compute_task_reward(self):
    forward_vel = self._robot.data.root_lin_vel_b[:, 0]
    target_vel = 4.0
    return torch.exp(-4.0 * (forward_vel - target_vel) ** 2)
```

⚠️ **为什么必须加速度奖励**：纯 AMP（task_w=0）只学风格不瞄速度，策略可能收敛到参考数据的任意速度。

#### Step 3：替换运动数据

```python
# 选项 A（推荐）: AMASS Retargeted for G1
#   HuggingFace: ember-lab-berkeley/AMASS_Retargeted_for_G1
#   已是 Isaac Lab AMP 兼容 .npz 格式

# 选项 B: 从 BeyondMimic .npz 转换
#   需写转换脚本匹配 _get_amp_observations() 的拼接顺序
#   ⚠️ 拼接顺序必须与 AMP 环境代码完全一致
```

#### Step 4：注册新任务（本地编辑 `__init__.py`）

```python
import gymnasium as gym

gym.register(
    id="RobotLab-Isaac-G1-AMP-Run-Direct-v0",
    entry_point="robot_lab.tasks.direct.amp....:G1AmpRunEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.g1_amp_run_env_cfg:G1AmpRunEnvCfg",
        "skrl_cfg_entry_point": f"{__name__}.agents.skrl_amp_cfg:...",
    },
)
```

在 `tasks/direct/__init__.py` 添加 import。

#### Step 5：训练（RunPod）

```bash
# git push → RunPod: git pull && pip install -e source/robot_lab

python scripts/reinforcement_learning/skrl/train.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 \
  --algorithm AMP --headless --num_envs 4096

# 评估
python scripts/reinforcement_learning/skrl/play.py \
  --task=RobotLab-Isaac-G1-AMP-Run-Direct-v0 \
  --algorithm AMP --num_envs 32
```

### 需要的修改

| 修改 | 量 | 说明 |
|------|---|------|
| 复制+改 env_cfg | ~20 行 | 加速度奖励，改权重 |
| 速度跟踪函数 | ~10 行 | `_compute_task_reward()` |
| 运动数据 | 替换或转换 | AMASS .npz 直接可用 |
| 任务注册 | ~10 行 | `__init__.py` |
| 判别器/网络 | **0** | 默认够用 |

### AMP 训练监控

| 指标 | 健康范围 | 异常处理 |
|------|---------|---------|
| disc_accuracy | 55-85% | >95% → 降 gradient_penalty 到 1.0 |
| forward_velocity | 趋近 4.0 | 停滞 → 增大 task_reward_weight |
| episode_length | 趋近 10s | <2s → 终止条件过严 |

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
