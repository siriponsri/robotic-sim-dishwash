# 03 — Environment & Task อธิบายละเอียด

> เอกสารนี้อธิบาย 2 custom environments, หุ่นยนต์ full-body, reward, contact

---

## สารบัญ

- [ภาพรวม 2 Tasks](#ภาพรวม-2-tasks)
- [Robot: Unitree G1 Full Body (37 DOF)](#robot-unitree-g1-full-body-37-dof)
- [Task 1: Apple Full-Body Env (Main)](#task-1-apple-full-body-env-main)
- [Task 2: DishWipe Full-Body Env (Bonus)](#task-2-dishwipe-full-body-env-bonus)
- [Observation & Action Space](#observation--action-space)
- [Balance & Safety](#balance--safety)

---

## ภาพรวม 2 Tasks

| Task | Env ID | Robot | ประเภท | NB |
|------|--------|-------|--------|-----|
| **Apple** (Main) | `UnitreeG1PlaceAppleInBowlFullBody-v1` | `unitree_g1` (37 DOF) | Pick-and-place | NB01–NB08 |
| **DishWipe** (Bonus) | `UnitreeG1DishWipeFullBody-v1` | `unitree_g1` (37 DOF) | Contact cleaning | NB09 |

ทั้ง 2 task ใช้ **full-body G1** (free-floating root, ต้องทรงตัว) — ไม่ใช่ upper body

---

## Robot: Unitree G1 Full Body (37 DOF)

### คุณสมบัติพื้นฐาน

| คุณสมบัติ | ค่า |
|----------|-----|
| Robot ID | `unitree_g1` |
| Class | `UnitreeG1` |
| URDF | `g1.urdf` |
| DOF | **37** |
| Root link | **Free-floating** (ต้องทรงตัวเอง) |
| fix_root_link | `False` |

### Joint Groups (37 DOF)

| Group | จำนวน | Joint Names |
|-------|-------|-------------|
| **Lower Body (ขา)** | 12 | `left/right_hip_pitch/roll/yaw_joint`, `left/right_knee_joint`, `left/right_ankle_pitch/roll_joint` |
| **Upper Body (ลำตัว+แขน)** | 11 | `torso_joint`, `left/right_shoulder_pitch/roll/yaw_joint`, `left/right_elbow_pitch/roll_joint` |
| **Left Hand** | 7 | `left_zero_joint` ... `left_six_joint` |
| **Right Hand** | 7 | `right_zero_joint` ... `right_six_joint` |

### Controller Config

| Parameter | ค่า |
|-----------|-----|
| control_mode | `pd_joint_delta_pos` |
| stiffness | 50 |
| damping | 1 |
| force_limit | 100 |
| action_space | `[-1, 1]^37` (normalized delta position) |

### Keyframes (ท่าเริ่มต้น)

G1 Full Body มี 3 keyframes ที่ตั้งไว้:
- **standing**: ขายืนตรง, แขนเอาไว้ข้างลำตัว
- **arms_up**: แขนชูขึ้น
- **squat**: ท่าย่อเข่า

### Balance Methods

```python
robot.is_standing()  # True ถ้ายืนตรงดี (root Z > threshold)
robot.is_fallen()    # True ถ้าล้ม (root Z ต่ำเกิน)
```

### เปรียบเทียบกับ Upper Body

| | Full Body (ใช้ในโปรเจกต์นี้) | Upper Body (original) |
|--|-------------------------------|----------------------|
| Robot ID | `unitree_g1` | `unitree_g1_simplified_upper_body` |
| DOF | **37** | **25** |
| Root | Free-floating (ต้อง balance) | Fixed (ลอยนิ่ง) |
| ขา | 12 joints (active) | ไม่มี |
| ความยาก | **สูง** (balance + manipulation) | ปานกลาง (manipulation เท่านั้น) |

---

## Task 1: Apple Full-Body Env (Main)

### คำอธิบาย

หุ่นยนต์ต้อง **หยิบแอปเปิล** จากเคาน์เตอร์ครัว แล้ว **วางลงในชาม** — ทั้งหมดนี้ขณะ **ทรงตัว** บนขาทั้งสอง

### Scene

- **Kitchen Counter** (KitchenCounterSceneBuilder, scale=0.82)
- **Apple**: actor ขนาดเล็ก, randomize ตำแหน่ง ±2.5 cm
- **Bowl**: kinematic actor, randomize ตำแหน่ง ±2.5 cm

### Reward Structure (Staged Dense Reward)

อ้างอิงจาก ManiSkill built-in `humanoid_pick_place.py`:

| Stage | Condition | Reward | Max |
|-------|-----------|--------|-----|
| **1. Reaching** | มือใกล้แอปเปิล | `1 - tanh(5 × dist(palm, apple))` | 1 |
| **2. Grasping** | Apple ถูกหยิบขึ้น | `2` (bonus on grasp) | 2 |
| **3. Placing** | Apple ใกล้ bowl | `1 - tanh(5 × dist(apple, bowl_center))` | 1 |
| **4. Release** | Apple อยู่ใน bowl + ปล่อย | `3` (final bonus) | 3 |
| **Standing bonus** | `is_standing()` | `+2` | 2 |
| **Fall penalty** | `is_fallen()` | `-10` + terminate | -10 |

> **Max reward per step ≈ 10** (after normalization)

### Success Condition

```python
success = apple_in_bowl AND gripper_open AND is_standing
```

### Custom vs Built-in

Built-in `UnitreeG1PlaceAppleInBowl-v1` ใช้ **upper body** (25 DOF, fixed root)
→ เราต้องสร้าง **custom env** ที่เปลี่ยน robot เป็น `unitree_g1` (37 DOF, free root)

ไฟล์: `src/envs/apple_fullbody_env.py` (TO CREATE)
- inherit จาก ManiSkill `HumanoidPlaceAppleInBowl`
- override `robot_uids` → `"unitree_g1"`
- เพิ่ม balance penalty/bonus ใน reward
- เพิ่ม `is_fallen()` termination

---

## Task 2: DishWipe Full-Body Env (Bonus)

### คำอธิบาย

หุ่นยนต์ต้อง **เช็ดจาน** ในอ่างล้างจาน ให้สะอาด ≥95% — ขณะทรงตัว

### Scene

- **Kitchen Counter** (เหมือน Apple task)
- **Sink** (อ่างล้างจาน) พร้อม plate
- **Plate**: 20cm × 20cm, kinematic actor

### Reward Structure (9-term Dense Reward)

อ้างอิงจาก `dishwipe_env.py` (ต้อง adapt เป็น full-body):

| Term | Weight | คำอธิบาย |
|------|--------|---------|
| `r_reach` | +1 | เข้าใกล้จาน |
| `r_contact` | +2 | สัมผัสจาน (multi-link contact) |
| `r_cleaning` | +10 | ทำความสะอาด cell ใหม่ |
| `r_progress` | +3 | cleaned_ratio เพิ่มขึ้น |
| `r_completion` | +50 | ล้างครบ ≥95% (bonus ครั้งเดียว) |
| `r_force` | -5 | soft force penalty (>50N) |
| `r_time` | -0.01 | ค่าปรับเวลา |
| `r_jerk` | -0.1 | action smoothness penalty |
| `r_act` | -0.01 | action magnitude penalty |
| **r_balance** | +2/-10 | **NEW**: standing bonus / fall penalty |

### VirtualDirtGrid

- Grid ขนาด 10×10 covering plate surface
- `brush_radius=1` → ทำความสะอาด 3×3 cells ต่อ contact point
- `world_to_cell()` — แปลง 3D position → grid (i, j)
- `mark_clean(i, j)` — mark cell + neighbors เป็น clean
- `get_cleaned_ratio()` → 0.0 ถึง 1.0

### Success Condition

```python
success = cleaned_ratio >= 0.95 AND is_standing
```

ไฟล์: `src/envs/dishwipe_fullbody_env.py` (TO CREATE)
- adapt จาก `dishwipe_env.py`
- เปลี่ยน robot เป็น `unitree_g1` (37 DOF)
- เพิ่ม balance reward/penalty

---

## Observation & Action Space

### Apple Full-Body Env

| Component | Dims | คำอธิบาย |
|-----------|------|---------|
| qpos | 37 | Joint positions |
| qvel | 37 | Joint velocities |
| tcp_pose | 7 | Left TCP position + quaternion |
| apple_pose | 7 | Apple position + quaternion |
| bowl_pose | 7 | Bowl position + quaternion |
| diff vectors | ~10 | palm→apple, apple→bowl distances |
| gripper state | 1 | Open/close |
| **Total obs** | **~110+** | Exact dim from NB01 smoke test |
| **Action** | **37** | pd_joint_delta_pos for all joints |

### DishWipe Full-Body Env

| Component | Dims | คำอธิบาย |
|-----------|------|---------|
| qpos | 37 | Joint positions |
| qvel | 37 | Joint velocities |
| tcp_pose | 7 | Left TCP position + quaternion |
| palm_pos | 3 | Palm link position |
| plate_pos | 3 | Plate center position |
| palm_to_plate | 3 | Direction vector |
| contact_force | 1 | Current contact force |
| cleaned_ratio | 1 | Progress (0-1) |
| dirt_grid | 100 | 10×10 grid state |
| **Total obs** | **~200+** | Exact dim from NB01 smoke test |
| **Action** | **37** | pd_joint_delta_pos for all joints |

---

## Balance & Safety

### Balance (ทั้ง 2 tasks)

Full-body G1 ต้องทรงตัว — ถ้าล้มจะ:
1. ได้ penalty `-10`
2. Episode terminate ทันที

```python
# ใน reward function
if robot.is_fallen():
    reward -= 10.0
    terminated = True
elif robot.is_standing():
    reward += 2.0  # standing bonus
```

### Safety Thresholds (DishWipe specific)

| Threshold | ค่า | Action |
|-----------|-----|--------|
| `FZ_SOFT` | 50 N | Soft penalty in reward |
| `FZ_HARD` | 200 N | Episode terminate |

> Apple task ใช้ staged reward ของ ManiSkill — ไม่มี explicit force threshold

---

*อัปเดตล่าสุด: มีนาคม 2026 | Full-Body G1 (37 DOF) — Apple + DishWipe*
