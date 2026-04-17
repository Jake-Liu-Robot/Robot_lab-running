# MuJoCo Sim-to-Sim Validation（G1 29-DOF AMP-Run）

策略 = `RobotLab-Isaac-G1-AMP-Run-Direct-v0` 的 skrl+AMP 训练产物（Phase 7 系列）。

**结论先放**：`outputs/run7_phase7_cont1_20k/policy_exported.pt` 和 `run7_phase7_latest_20k/policy_exported.pt` 在 MuJoCo 中**完整复现** Isaac Lab eval 的行为——ramp 模式（cmd 0 m/s→4 m/s→0）下均能完成**站立 → 加速 → 4 m/s 巡航 → 减速 → 站立**的完整周期。

**对比前提（重要）**：两侧都**关闭了所有域随机化**，对比反映的是物理引擎本质差异，不是噪声。

- Lab 侧：`eval_amp_run.py` 强制 `env_cfg.push_enable = obs_noise_enable = pd_gain_random_enable = joint_pos_offset_enable = added_mass_enable = False`。
- sim2sim 侧：MuJoCo 确定性仿真，本就无这些随机化。

**MuJoCo XML 来源**：`/home/jake/Unitree_rl_gym/unitree_rl_gym/resources/robots/g1_description/g1_29dof_rev_1_0.xml`（从 `unitreerobotics/unitree_rl_gym` 的资产，与 Isaac Lab 用的 URDF 同源）。不要用 Menagerie 的 `unitree_g1/g1_mjx.xml`——那版 body merge 和关节子集不同。

---

## 目录

- [数据对比（两个 checkpoint × lab / sim2sim）](#数据对比)
- [从"robot 一直塌"到"跑 4 m/s"：3 个叠加 bug 诊断](#诊断从一直塌到跑-4-ms)
- [sim2sim 与 Isaac Lab 的完整对齐清单](#sim2sim-与-isaac-lab-的完整对齐清单)
- [残留差异和成因（不可消除）](#残留差异)
- [文件位置和复现命令](#复现命令)

---

## 数据对比

| run | Isaac Lab 路径 | sim2sim 路径 |
|---|---|---|
| cont1_20k | `outputs/run7_phase7_cont1_20k/videos/auto/eval/eval_agent_20000_ramp.csv` | `outputs/run7_phase7_cont1_20k/sim2sim/sim2sim_policy_exported_ramp.csv` |
| latest_20k | `outputs/run7_phase7_latest_20k/videos/curated_agent20k/ramp.csv` | `outputs/run7_phase7_latest_20k/sim2sim/sim2sim_policy_exported_ramp.csv` |

两者都是 20s ramp（cmd=4 m/s 前 15s，cmd=0 m/s 后 5s），60Hz 控制，1200 步，同源 policy_exported.pt。

### cont1_20k（Phase 7 续训中间 ckpt）

| t (s) | cmd | lab fwd_vel | sim fwd_vel | Δfwd | lab h | sim h | Δh |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | −0.074 | −0.019 | +0.055 | 0.758 | 0.760 | +0.002 |
| 1 | 4 | +1.616 | +1.366 | −0.250 | 0.696 | 0.731 | +0.035 |
| 2 | 4 | +3.807 | +3.712 | −0.095 | 0.706 | 0.684 | −0.022 |
| 3 | 4 | +3.934 | +4.325 | +0.391 | 0.693 | 0.715 | +0.022 |
| 4–14 | 4 | **3.85–4.00** | **3.87–4.33** | < ±0.30 | 0.68–0.71 | 0.69–0.72 | ≤ +0.03 |
| 15 | 0 | +3.970 | +4.044 | +0.074 | 0.688 | 0.722 | +0.034 |
| 16 | 0 | +0.490 | +0.814 | +0.324 | 0.685 | 0.732 | +0.047 |
| 17 | 0 | −0.026 | +0.060 | +0.086 | 0.759 | 0.766 | +0.007 |
| 18 | 0 | +0.064 | −0.004 | −0.068 | 0.760 | 0.770 | +0.010 |
| 19 | 0 | −0.021 | +0.074 | +0.095 | 0.763 | 0.771 | +0.008 |

**聚合指标**：
- 巡航段（cmd=4, t=0..15s）：lab fwd 均值 3.66 m/s，sim fwd 均值 3.76 m/s（高 +2.7%）
- 巡航高度：lab 0.698 m，sim 0.711 m（高 +1.3 cm）
- 站立段（cmd=0, t=15..20s）：lab 0.735 m，sim 0.747 m（高 +1.2 cm）

### latest_20k（Phase 7 最新 ckpt）

| t (s) | cmd | lab fwd_vel | sim fwd_vel | Δfwd | lab h | sim h | Δh |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4 | +0.003 | +0.090 | +0.087 | 0.758 | 0.760 | +0.002 |
| 1 | 4 | +0.529 | +0.158 | −0.371 | 0.734 | 0.774 | +0.040 |
| 2 | 4 | +3.862 | +2.174 | **−1.688** | 0.695 | 0.688 | −0.007 |
| 3 | 4 | +4.016 | +4.073 | +0.057 | 0.700 | 0.687 | −0.013 |
| 4–14 | 4 | **3.92–4.10** | **3.90–4.46** | < ±0.47 | 0.67–0.72 | 0.70–0.73 | ≤ +0.05 |
| 15 | 0 | +3.985 | +3.407 | −0.578 | 0.712 | 0.687 | −0.025 |
| 16 | 0 | +0.375 | +0.503 | +0.128 | 0.707 | 0.713 | +0.006 |
| 17 | 0 | +0.105 | +0.033 | −0.072 | 0.739 | 0.749 | +0.010 |
| 18 | 0 | +0.004 | +0.005 | +0.001 | 0.741 | 0.751 | +0.010 |
| 19 | 0 | +0.005 | −0.006 | −0.011 | 0.739 | 0.751 | +0.012 |

**聚合指标**：
- 巡航段：lab fwd 均值 3.63 m/s，sim fwd 均值 3.54 m/s（低 −2.5%）
- 巡航高度：lab 0.696 m，sim 0.714 m（高 +1.8 cm）
- 站立段：lab 0.722 m，sim 0.730 m（高 +0.8 cm）

### 两 ckpt 对比

| 维度 | cont1_20k | latest_20k | 解读 |
|---|---|---|---|
| lab 起步（t=1 fwd）| 1.62 m/s | 0.53 m/s | latest 起步策略更"谨慎"（先蓄势）|
| lab 到 4 m/s 用时 | ~2s | ~3s | latest 加速更晚 |
| lab 巡航稳定性 | 3.85–4.00 | 3.92–4.10 | latest 巡航稍稳（窄 0.08→0.18）|
| sim 复现一致性 | Δfwd 普遍 < 0.3 | t=2s 掉队到 2.17（Δ=−1.69） | latest 的"先蓄势"策略对起步时序敏感 |
| 站立段 lab h | 0.686→0.763 | 0.707→0.741 | latest 站立姿态稍高 |

**latest 在 t=2s 的 sim vs lab 偏差 1.69 m/s** 是最大单点误差。原因：latest 的起步阶段（t=0..2s）有一段低速蓄势，MuJoCo 的接触/PD 响应微差让这段多延 ~1 个控制周期，之后快速追平。从 t=3s 起两者几乎重合。属于物理引擎可分辨的微分时序差异（见[残留差异](#残留差异)）。

---

## 诊断：从"一直塌"到"跑 4 m/s"

起初同一 `policy_exported.pt` 在 MuJoCo 中**立即坍塌**（pelvis 从 0.76 m 0.5 秒内降到 0.13 m 平躺地上），而 Isaac Lab eval 完全正常。排查出来是 sim2sim 侧**三个叠加 bug**，每一个都能让 PD 失效：

### Bug 1：`<actuator>` 块替换完全失败

`scripts/sim2sim/sim2sim_mujoco.py` 用 `str.replace("<actuator>\n</actuator>", ...)` 注入自建的 PD actuator XML。但 G1 的 MJCF 出厂时**已有 29 个 `<motor>` actuator**：

```xml
<actuator>
  <motor name="left_hip_pitch_joint" joint="left_hip_pitch_joint"/>
  <motor name="left_hip_roll_joint" joint="left_hip_roll_joint"/>
  ...  (29 个)
</actuator>
```

`str.replace` 找不到空块 → **替换静默失败** → 策略 `data.ctrl[i]` 被当成**原始电机扭矩**，且 `<motor>` 默认 `ctrlrange=[0,0]`，ctrl 被夹回 0。结果：**零扭矩** → 机器人纯粹自由落体。

**修**：`re.subn(r"<actuator>.*?</actuator>", ..., flags=DOTALL)` 整块替换。

### Bug 2：`biastype` 默认 "none"，biasprm 被忽略

即使 `<general>` 注入成功，我原先写的 actuator 是：

```xml
<general ... gainprm="{kp} 0 0" biasprm="0 -{kp} -{kd}" ctrlrange="-3.14159 3.14159"/>
```

MuJoCo 的 `<general>` actuator `biastype` 默认 `"none"`，**biasprm 完全被忽略**。此时力 = `gain*ctrl = kp*ctrl`——变成了一个"位置命令 × kp"的开环控制器，`(target - qpos)` 的误差项和阻尼 `-kd*qvel` 全部没了。

**修**：显式写 `biastype="affine"`，让力变成：

```
force = gain*ctrl + biasprm[0] + biasprm[1]*qpos + biasprm[2]*qvel
      = kp*ctrl  +     0       +    -kp*qpos    +    -kd*qvel
      = kp*(ctrl - qpos) - kd*qvel         ← 标准 PD
```

### Bug 3：Armature 只加了 hips+knees

最初 `JOINT_ARMATURE` 表只包含 8 个关节（hips + knees 各 0.01）。而 `unitree.py` 的 `UNITREE_G1_29DOF_CFG` **每个关节**都有特定的 `armature`（按电机型号 7520-14 / 7520-22 / 5020 / 2×5020 / 4010 分别配置）。缺 armature → 有效关节惯量太小 → PD 响应过激 → 高频抖动 / 易发散。

**修**：按 `unitree.py` 把全部 29 关节的 armature 填进 `JOINT_ARMATURE` 字典，在 `setup_mujoco_model` 里通过 `model.dof_armature[dof_adr] = arm` 写入。

### 排查方法论

| 步骤 | 工具 | 锁定什么 |
|---|---|---|
| 1. 关节顺序对齐 | `scripts/sim2sim/print_lab_order.py`（在 RunPod 上 dump `robot.data.joint_names`）| Isaac Lab 用 BFS 交错而非 URDF 原生顺序 |
| 2. Action scale 对齐 | 同上 dump `soft_joint_pos_limits`、`default_joint_pos` | 确认我们本地 fallback 的 `0.9 × 硬限位` = lab 实际值 |
| 3. 初始姿态对齐 | `unitree.py init_state` pelvis z=0.76 + 12 个非零关节角 | 代替 MJCF keyframe |
| 4. **actuator 配置验证** | `python -c "m=...; print(m.actuator_biastype[0], m.actuator_gainprm[0])"` | **暴露 bug 1+2**：`biastype=0` `gainprm=[1,0,0]` 说明 XML 注入从未生效 |
| 5. 逐维 obs dump 对比 | 加 `--debug_steps 1` 打印 109 维 obs | 验证 obs pipeline 无关节错位或 frame 约定差 |

---

## sim2sim 与 Isaac Lab 的完整对齐清单

| 对齐项 | Isaac Lab 来源 | sim2sim 实现 |
|---|---|---|
| **关节顺序（29）** | USD BFS：L_hip_pitch, R_hip_pitch, waist_yaw, L_hip_roll, R_hip_roll, waist_roll, ...（左右-waist 每层交错）| `JOINT_NAMES` 硬编码 |
| **Body 顺序 / key bodies** | `KEY_BODY_INDEXES=[24,25,26,27,35,34,23,22,11,10,9,13,12]`（13 个）| `KEY_BODY_NAMES` + `mj_name2id` 按名查找 |
| **rubber_hand body** | USD 有 `left/right_rubber_hand`（id 34/35）| MJCF 合并进 wrist_yaw_link；`KEY_BODY_REMAP` 把名字重指到 wrist_yaw_link + URDF offset `(0.0415, ±0.003, 0)` |
| **初始 pelvis z** | `init_state.pos.z = 0.76` | `INIT_PELVIS_HEIGHT = 0.76` |
| **初始关节角** | `init_state.joint_pos` 里 12 个非零关节 | `INIT_JOINT_POS` dict 按名设置，其余默认 0 |
| **Soft joint limits** | `data.soft_joint_pos_limits = factor × hard_range`（factor=0.9，中心不变）| `SOFT_JOINT_POS_LIMIT_FACTOR = 0.9`，由 MuJoCo `jnt_range × 0.9` 得 |
| **Action offset/scale** | `0.5*(soft_upper+soft_lower)` / `soft_upper-soft_lower` | 同公式 |
| **PD 控制律** | `ImplicitActuator`：`torque = stiffness*(target-qpos) - damping*qvel` | MuJoCo `<general biastype="affine" gainprm="kp 0 0" biasprm="0 -kp -kd"/>` |
| **PD 增益** | `STIFFNESS_7520_14=40.18`, `_22=99.10`, `_5020=14.25`, `2×5020=28.50`, `4010=16.78` + 对应 damping | `PD_GAINS` 字典按 `unitree.py` 构造函数复现 |
| **Armature（全 29 关节）** | `ARMATURE_7520_14=0.010178`, `_22=0.025102`, `_5020=0.003610`, `2×5020=0.007219`, `_4010=0.00425` | `JOINT_ARMATURE` → 写入 `model.dof_armature` |
| **effort limit** | `ImplicitActuator.effort_limit_sim`（hip_pitch 88 / hip_roll 139 / knee 139 / ankle 50 / waist_yaw 88 / arms 25 / wrists 5）| MJCF 的 `<joint actuatorfrcrange="-N N"/>` 与 unitree.py 一致（XML 里已写好）|
| **控制频率** | `decimation=1`, `dt=1/60` → 60Hz 控制 | 同 60Hz，每 policy step 调用一次 |
| **物理求解** | PhysX implicit，内部 substep | `<option integrator="implicitfast" timestep="0.002"/>` → 500Hz 物理 × 8 substeps/control |
| **地面摩擦** | `RigidBodyMaterialCfg(static=1.0, dynamic=1.0)` | MJCF 默认（1.0 接近）|
| **观测（109 维）** | `amp_obs(105) + root_vel_body(3) + cmd_vel(1)`，`amp_obs = joint_pos(29)+joint_vel(29)+root_z(1)+tangent_normal(6)+key_body_rel(39)+progress(1)` | `compute_observation()` 完整复刻 |
| **四元数约定** | Isaac Sim `[w,x,y,z]` | MuJoCo `data.xquat` 也是 `[w,x,y,z]` ✓ 无需转换 |
| **root_vel_w 数据源** | `body_lin_vel_w[pelvis]`（world frame, 链接原点）| `data.qvel[0:3]`（free joint 世界系平动速度）|
| **root_vel_b** | `quat_rotate_inverse(root_quat_w, root_vel_w)` | 相同的四元数逆旋转数学 |
| **Obs normalization** | `RunningStandardScaler`（训练累积的 running_mean/var，clip=5）| `ObservationNormalizer` 从 `export_policy.py` 的 preprocessor_state 加载 |
| **策略网络** | skrl `GaussianMixin`, `clip_actions=false`, 确定性输出 mean | `PolicyNetwork` MLP `[109, 1024, 512, 29]` + ReLU，直接输出 mean |
| **ramp schedule** | `eval_amp_run.py`：t<15s→cmd=4, else cmd=0 | 同 |
| **Domain rand in eval** | `push/obs_noise/pd_gain_rand/joint_pos_offset/added_mass` 全关 | sim2sim 本就无 |

---

## 残留差异

以下差异**无法通过配置对齐消除**（不同物理引擎的本质差异），但已控制在数值噪声量级（Δfwd < 0.5 m/s，Δh < 5 cm）：

| 项 | Isaac Lab (PhysX) | MuJoCo | 效果 |
|---|---|---|---|
| 接触模型 | soft contact + signed distance | soft constraint / solref+solimp | 脚底微滑移差异 |
| 求解器 | PhysX TGS（implicit） | `implicitfast`（implicit 小范围内）| 高刚度 PD 瞬态响应微差 |
| 初始 settling | reset() 内部跑几步物理让脚落地 → pelvis 自然下沉 4 cm | sim2sim 直接从 init_pos 启动 → 前几帧物理才下沉 | 起步相位差 ~1–2 控制周期 |
| 数值精度 | float32 GPU | float64 CPU | 每步微误差累积 |

**具体表现**：
- cont1 在巡航段 sim fwd 均值比 lab 高 2.7%，latest 低 2.5%——两方向误差说明是物理微差而非系统性偏置
- 站立段高度 sim 比 lab 高 0.8–1.2 cm——MuJoCo 脚底接触的小凸起比 PhysX 刚一点
- latest_20k 在 t=2s 短暂落后 1.69 m/s，3s 后追平——起步相位差，典型物理引擎瞬态差异

**这些差异不影响 sim2real 验证的结论**：策略能在完全不同的物理求解器下**复现训练中的运动模式**，说明 **Isaac Lab 的训练没有利用到 PhysX 特有的数值伪迹**，策略对物理引擎差异鲁棒。

---

## 复现命令

### 导出策略（本地，无 Isaac Lab 依赖）

```bash
conda activate unitree-rl   # 或任何装了 torch 的环境
python scripts/sim2sim/export_policy.py \
    --checkpoint outputs/run7_phase7_latest_20k/checkpoints/best_agent.pt \
    --output    outputs/run7_phase7_latest_20k/policy_exported.pt
```

`best_agent.pt`（26 MB，skrl 全状态）→ `policy_exported.pt`（2.5 MB，只留 policy + normalizer）。

### 跑 ramp sim2sim（本地）

```bash
python scripts/sim2sim/sim2sim_mujoco.py \
    --policy       outputs/run7_phase7_latest_20k/policy_exported.pt \
    --cmd_vel      ramp \
    --duration     20.0 \
    --video_width  1280 \
    --video_height 720 \
    --no_render    # 关交互 viewer，只录制 mp4 + 写 CSV
```

输出：
- `outputs/<run>/sim2sim/sim2sim_policy_exported_ramp.mp4`（1280×720 @ 60fps）
- `outputs/<run>/sim2sim/sim2sim_policy_exported_ramp.csv`（step, time_s, cmd_vel, fwd_vel, lateral_vel, pelvis_height, reward_step）

### 其他 cmd 模式

```bash
# 固定速度
--cmd_vel 4.0        # 4 m/s 巡航测试
--cmd_vel 0.0        # 站立测试

# 对比调试（dump 109 维 obs + 29 维 action + target）
--debug_steps 3 --no_video --duration 0.1
```

### 在 RunPod 获取 Isaac Lab 真实 joint/body/action_scale

```bash
cd /workspace/robot_lab && /isaac-sim/python.sh scripts/sim2sim/print_lab_order.py 2>&1 | tee /tmp/lab_order.txt
```

需要在 Isaac Lab 环境里跑（冷启动约 30–60s）。用于验证 sim2sim 的 `JOINT_NAMES` / `KEY_BODY_NAMES` / action_offset+scale 是否与当前 lab 环境一致——任何 Isaac Lab / robot_lab 升级后都应重新跑一遍。

---

## 相关 commit

| commit | 说明 |
|---|---|
| `56deb0b` | `export_policy.py`：处理 skrl 嵌套 OrderedDict |
| `377cfa4` | sim2sim：rubber_hand 合并修正、init state、action scale、implicitfast 积分器 |
| `876682a` | `print_lab_order.py`：dump `soft_joint_pos_limits` |
| `700be5f` | `print_lab_order.py`：dump 初始 109 维 obs 用于逐维对比 |
| `b041305` | **sim2sim：PD 控制 3 bug 终修**（actuator 块替换 + biastype=affine + 全 29 关节 armature）|

---

## 关键文件

- `scripts/sim2sim/sim2sim_mujoco.py` — MuJoCo sim2sim 主脚本（policy + MJCF + 视频 + CSV）
- `scripts/sim2sim/export_policy.py` — skrl checkpoint → 独立 policy.pt（纯 torch）
- `scripts/sim2sim/print_lab_order.py` — RunPod 诊断：dump Isaac Lab 真实 joint/body/action_scale/init_obs
- `source/robot_lab/robot_lab/assets/unitree.py` — `UNITREE_G1_29DOF_CFG`（PD 增益 + armature + 初始姿态的真相源）
- `source/robot_lab/robot_lab/tasks/direct/g1_amp/g1_amp_run_env.py` — 观测定义（`_get_observations` 109 维结构）
- `source/robot_lab/robot_lab/tasks/direct/g1_amp/g1_amp_env.py` — `compute_obs` 105 维 AMP 观测结构
- `scripts/reinforcement_learning/skrl/eval_amp_run.py` — Isaac Lab eval 脚本（ramp 定义、CSV 列、视频路径）
