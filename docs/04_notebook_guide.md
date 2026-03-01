# 04 — คู่มือ Notebook (NB01–NB09) แบบละเอียด

> เอกสารนี้อธิบายทุก notebook อย่างละเอียด: ทำอะไร, ต้องมีอะไรก่อน, ได้อะไรออกมา, ปัญหาที่อาจเจอ

---

## สารบัญ

- [Dependency Graph](#dependency-graph)
- [NB01 — Setup & Smoke Test](#nb01--setup--smoke-test)
- [NB02 — Grid Mapping](#nb02--grid-mapping)
- [NB03 — Dirt Engine](#nb03--dirt-engine)
- [NB04 — Reward & Safety + MLflow](#nb04--reward--safety--mlflow)
- [NB05 — Baselines & Smoothing](#nb05--baselines--smoothing)
- [NB06 — Train PPO](#nb06--train-ppo)
- [NB07 — Train SAC](#nb07--train-sac)
- [NB08 — Residual SAC + Ablation](#nb08--residual-sac--ablation)
- [NB09 — Evaluation & Summary](#nb09--evaluation--summary)
- [สรุปตาราง I/O ทุก NB](#สรุปตาราง-io-ทุก-nb)

---

## Dependency Graph

```mermaid
graph TD
    NB01["<b>NB01</b><br/>Setup & Smoke Test<br/>📁 NB01_setup_smoke.ipynb"]
    NB02["<b>NB02</b><br/>Grid Mapping<br/>📁 NB02_grid_mapping.ipynb"]
    NB03["<b>NB03</b><br/>Dirt Engine<br/>📁 NB03_dirt_engine.ipynb"]
    NB04["<b>NB04</b><br/>Reward & Safety<br/>📁 NB04_reward_safety_mlflow.ipynb"]
    NB05["<b>NB05</b><br/>Baselines & Smoothing<br/>📁 NB05_baselines_smoothing.ipynb"]
    NB06["<b>NB06</b><br/>Train PPO 🔥GPU<br/>📁 NB06_train_ppo.ipynb"]
    NB07["<b>NB07</b><br/>Train SAC 🔥GPU<br/>📁 NB07_train_sac.ipynb"]
    NB08["<b>NB08</b><br/>Residual SAC 🔥GPU<br/>📁 NB08_residual_sac_ablation.ipynb"]
    NB09["<b>NB09</b><br/>Evaluation 📊<br/>📁 NB09_evaluation.ipynb"]

    NB01 --> NB02
    NB01 --> NB03
    NB01 --> NB04
    NB02 --> NB05
    NB03 --> NB05
    NB04 --> NB05
    NB05 --> NB06
    NB05 --> NB07
    NB05 --> NB08
    NB07 -.->|"SAC hyperparams"| NB08
    NB06 --> NB09
    NB07 --> NB09
    NB08 --> NB09

    style NB01 fill:#4CAF50,color:#fff
    style NB02 fill:#4CAF50,color:#fff
    style NB03 fill:#4CAF50,color:#fff
    style NB04 fill:#4CAF50,color:#fff
    style NB05 fill:#81C784,color:#000
    style NB06 fill:#FF9800,color:#fff
    style NB07 fill:#FF9800,color:#fff
    style NB08 fill:#FF9800,color:#fff
    style NB09 fill:#2196F3,color:#fff
```

---

## NB01 — Setup & Smoke Test

📁 `notebooks/NB01_setup_smoke.ipynb`

### Objective (ทำไมต้องมี)
ตรวจสอบว่าทุกอย่างพร้อมก่อนเริ่มทำงาน:
- Python version, OS, libraries ครบ
- Environment สร้างได้
- Observation/Action shape ถูกต้อง
- Contact API ทำงาน
- Dirt grid ทำงาน
- MLflow เชื่อมต่อได้

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| Virtual environment (`.env/`) | ขั้นตอน setup |
| `src/envs/dishwipe_env.py` | source code |
| `src/envs/dirt_grid.py` | source code |
| `.env.local` (MLflow credentials) | สร้างเอง |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `env_spec.json` | `artifacts/NB01/` | Obs/act shape, env_id, robot info |
| `active_joints.json` | `artifacts/NB01/` | รายชื่อ 25 active joints |
| `requirements.txt` | `artifacts/NB01/` | pip freeze snapshot |
| `contact_test.json` | `artifacts/NB01/` | Contact API test result |
| `nb01_config.json` | `artifacts/NB01/` | Config ที่ใช้ |
| `smoke_video.mp4` | `artifacts/NB01/` | (optional — ถ้า render ได้) |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "env_id": "UnitreeG1DishWipe-v1",
    "control_mode": "pd_joint_delta_pos",
    "obs_mode": "state",
    "smoke_steps": 50,
}
```

### Expected Runtime
- **CPU**: 1–2 นาที
- **GPU**: 30 วินาที

### 📊 Artifact Interpretation (ตีความผลที่ได้)

หลังรัน NB01 จะได้ artifacts ใน `artifacts/NB01/`:

**`env_spec.json`** — ข้อมูล environment:
```json
{
  "obs_space_shape": [1, 168],   // obs 168 มิติ ✅
  "act_space_shape": [25],       // action 25 ข้อต่อ ✅
  "robot_dof": 25,
  "scene": "kitchen_counter + sink + plate"
}
```
> 🔍 **แปลผล**: obs = 168 = qpos(25) + qvel(25) + base(~4) + tcp(7) + palm(3) + plate(3) + vec(3) + force(1) + ratio(1) + grid(100) — ถ้าไม่ใช่ 168 แสดงว่า env version ไม่ตรง

**`smoke_summary.json`** — ผลทดสอบ 50 steps ด้วย random policy:
```json
{
  "mean_reward": -0.0059,     // ติดลบเล็กน้อย (ปกติ — time+jerk penalty)
  "max_contact_force": 0.0,   // random ไม่แตะจานเลย
  "final_cleaned_ratio": 0.0  // ยังไม่มีอะไรสะอาด
}
```
> 🔍 **แปลผล**: reward ≈ -0.006 ต่อ step เพราะ `r_time = -0.01` + `r_jerk + r_act` ถูกลบ ขณะที่ `r_reach ≈ 0.005` (ห่างจากจาน) — ปกติ 100% สำหรับ random policy

**`tcp_info.json`** — ตำแหน่ง TCP และ Palm:
```json
{
  "left_palm_pos": [-0.097, 0.159, 0.841],  // palm อยู่สูง ~0.84m
  "left_tcp": [-0.027, 0.125, 0.843]         // tcp offset ~7cm จาก palm
}
```
> 🔍 **แปลผล**: palm อยู่ที่ z=0.84 ขณะที่จานอยู่ z=0.58 → ต้องเอามือลงไป ~26cm — นี่คือสิ่งที่ `r_reach` กระตุ้น

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่ต้องได้ | ถ้าไม่ตรง |
|-------|-------------|----------|
| obs shape | `[1, 168]` | env version ผิด |
| act shape | `(25,)` | robot ผิดรุ่น |
| smoke_passed | `true` | dependency ขาด |
| mean_reward | ≈ -0.006 (ช่วง -0.009 ถึง -0.004) | reward function เปลี่ยน |
| contact_force | 0.0 (random ไม่แตะจาน) | ปกติ |
| active joints | 25 อัน | URDF เปลี่ยน |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| `ModuleNotFoundError: No module named 'src'` | path ผิด | ตรวจว่า `sys.path.insert(0, PROJECT_ROOT)` ทำงาน |
| Render failed (Vulkan error) | ไม่มี GPU | ปกติ — มี try/except แล้ว ไม่กระทบผล |
| Obs shape ≠ 168 | env version ผิด | ตรวจว่า import `UnitreeG1DishWipeEnv` ก่อน `gym.make()` |

---

## NB02 — Grid Mapping

📁 `notebooks/NB02_grid_mapping.ipynb`

### Objective
ทดสอบระบบ mapping ระหว่าง world coordinate กับ dirt grid:
- แปลง world position → UV → cell index ถูกต้อง
- Contact detection ทำงาน
- zig-zag path ครอบคลุมทุก cell
- Visualization ของ grid before/after

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB01 ผ่าน (env ใช้ได้) | NB01 |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `grid_before.png` | `artifacts/NB02/` | Grid ก่อน cleaning |
| `grid_after.png` | `artifacts/NB02/` | Grid หลัง zig-zag clean |
| `grid_trace.csv` | `artifacts/NB02/` | zig-zag waypoint trace |
| `env_exploration_trace.csv` | `artifacts/NB02/` | Random exploration trace |
| `nb02_config.json` | `artifacts/NB02/` | Config |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "grid_h": 10, "grid_w": 10,
    "brush_radius": 1,
    "sim_steps": 100,
    "n_explore": 200,
}
```

### Expected Runtime
- **CPU**: 1–2 นาที

### 📐 Algorithm: World → Grid Coordinate Mapping

NB02 ทดสอบสมการแปลงพิกัดจาก 3D world space → 2D grid cell:

$$u = \frac{x - (c_x - h_x)}{2 \cdot h_x}, \quad v = \frac{y - (c_y - h_y)}{2 \cdot h_y}$$

$$i = \lfloor v \times H \rfloor, \quad j = \lfloor u \times W \rfloor$$

โดย:
- $(c_x, c_y)$ = ตำแหน่งกลางจาน (0.10, 0.20)
- $(h_x, h_y)$ = half-size ของจาน (0.10, 0.10)
- $H = W = 10$ (grid dimensions)

> ตัวอย่าง: ถ้ามือสัมผัสที่ world $(0.15, 0.25, 0.58)$ → $u = 0.75, v = 0.75$ → cell $(7, 7)$

### 📊 Artifact Interpretation

**`grid_trace.csv`** — zig-zag path ที่ NB02 สร้างเพื่อทดสอบ:
```csv
step, u,     v,     cell_i, cell_j, delta_clean, cleaned_ratio
0,    0.050, 0.050, 0,      0,      4,           0.04
1,    0.150, 0.050, 0,      1,      2,           0.06
...
99,   0.050, 0.950, 9,      0,      0,           1.00
```
> 🔍 **แปลผล**: step 0 ได้ `delta_clean=4` เพราะ brush radius=1 ล้าง 3×3 แต่ขอบ grid ตัดเหลือ 4 cells, cleaned_ratio เพิ่มจนครบ 100% (100/100 cells) — ยืนยันว่า mapping ถูกต้อง

**`grid_before.png` / `grid_after.png`** — ภาพ grid ก่อน-หลัง:
- **Before**: grid 10×10 สีแดงทั้งหมด (dirty)
- **After**: grid 10×10 สีเขียวทั้งหมด (clean) — ยืนยัน zig-zag ครอบคลุมทุก cell

**`env_exploration_trace.csv`** — random policy 100 steps:
```csv
step, tcp_x,   tcp_y,   tcp_z,   contact_force, cleaned_ratio
0,    -0.045,  0.165,   0.837,   0.000,         0.000
...
```
> 🔍 **แปลผล**: random policy ไม่ contact จานเลย (force=0 ทุก step, cleaned=0%) — เพราะ TCP อยู่สูง ~0.84m แต่จานอยู่ 0.58m → random ไม่มีทางไปถึง → ยืนยันว่าต้องมี RL policy ที่เรียนรู้

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่ต้องได้ |
|-------|-------------|
| zig-zag coverage | **100/100 cells** (1.00) |
| Random contact rate | **0%** (ปกติ) |
| Plate center from trace | ≈ (0.09–0.11, 0.19–0.21, 0.58) |
| grid_after.png | สีเขียวทั้งหมด |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| Corner mapping ผิด | plate position เปลี่ยน | ตรวจ `PLATE_POS_IN_SINK` |
| Contact rate = 0% | Random action ไม่สัมผัสจาน | ปกติ — random ไม่ฉลาดพอ |

---

## NB03 — Dirt Engine

📁 `notebooks/NB03_dirt_engine.ipynb`

### Objective
ทดสอบ VirtualDirtGrid อย่างทั่วถึง:
- Brush radius effect (r=0, 1, 2)
- Cleaning progress tracking
- Coordinate mapping validation

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB01 ผ่าน (env ใช้ได้) | NB01 |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `brush_effect_demo.png` | `artifacts/NB03/` | Visualization ของ brush radius ต่างๆ |
| `cleaning_trace.csv` | `artifacts/NB03/` | ข้อมูล cleaning per step |
| `nb03_config.json` | `artifacts/NB03/` | Config |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "grid_h": 10, "grid_w": 10,
    "brush_radii_to_test": [0, 1, 2],
    "env_test_steps": 50,
}
```

### Expected Runtime
- **CPU**: 1 นาที

### 📐 Algorithm: Brush Effect (Cleaning Footprint)

NB03 ทดสอบ brush effect — เมื่อสัมผัส cell $(i, j)$ จะล้าง cells รอบข้างด้วย:

$$\text{cleaned\_cells} = \{(i+di, j+dj) \mid |di| \leq r, |dj| \leq r, 0 \leq i+di < H, 0 \leq j+dj < W\}$$

เมื่อ $r$ = `brush_radius` → footprint = $(2r+1) \times (2r+1)$:

| Brush Radius | Footprint | Max cells/touch |
|:---:|:---:|:---:|
| 0 | 1×1 | 1 |
| **1 (default)** | **3×3** | **9** |
| 2 | 5×5 | 25 |

> ⚠️ ที่ขอบ grid จะได้ cells น้อยกว่า max เพราะถูก clip (เช่น cell (0,0) ที่ r=1 ได้ 4 cells แทน 9)

### 📊 Artifact Interpretation

**`brush_effect_demo.png`** — ภาพเปรียบเทียบ brush 3 ขนาด:
- **r=0**: สัมผัส 1 cell ได้ 1 cell สะอาด (จุดเดียว)
- **r=1**: สัมผัส 1 cell ได้ 9 cells สะอาด (กรอบ 3×3)
- **r=2**: สัมผัส 1 cell ได้ 25 cells สะอาด (กรอบ 5×5)

**`cleaning_trace.csv`** — random policy 50 steps:
```csv
step, delta_clean, cleaned_ratio
0,    0.0,         0.0
...
49,   0.0,         0.0
```
> 🔍 **แปลผล**: `delta_clean = 0` ทุก step → random policy ไม่สัมผัสจาน → ยืนยันว่า cleaning ต้องมี contact จริง (ไม่มี bug ล้างลอย) — standalone unit test ข้างบนยืนยันว่า grid ทำงานถูกต้อง

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่ต้องได้ |
|-------|-------------|
| Unit test (standalone brush) | ทุก radius ล้างจำนวน cell ถูกต้อง |
| brush r=1 ตรงกลาง | delta_clean = 9 |
| brush r=1 ที่มุม (0,0) | delta_clean = 4 |
| Random policy cleaning | 0% (ปกติ) |
| brush_effect_demo.png | แสดง 3 ภาพ r=0/1/2 |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| `delta_clean = 0` ทุก step | Random action ไม่ contact | ปกติ — standalone test ยืนยัน grid ทำงานได้ |

---

## NB04 — Reward & Safety + MLflow

📁 `notebooks/NB04_reward_safety_mlflow.ipynb`

### Objective
- สร้าง **reward contract** (เอกสาร 9 terms + safety rules) เป็น JSON
- ทดสอบ safety termination conditions
- สร้าง MLflow helper functions สำหรับ NB ถัดไป
- รัน test episodes ด้วย random policy แล้วตรวจ reward range

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB01 ผ่าน (env ใช้ได้) | NB01 |
| `.env.local` (MLflow creds) | สร้างเอง |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `reward_contract.json` | `artifacts/NB04/` | Reward function spec ทั้งหมด |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "test_episodes": 5,
    "test_steps_per_ep": 100,
}
```

### Expected Runtime
- **CPU**: 1–2 นาที

### 📐 Algorithm: 9-Term Staged Reward

NB04 สร้าง **reward contract** ที่เป็นเอกสารอ้างอิงของ reward function ที่ใช้ในการเทรน:

$$r_t = \underbrace{r_{\text{reach}} + r_{\text{contact}} + r_{\text{clean}} + r_{\text{sweep}}}_{\text{positive}} + \underbrace{r_{\text{time}} + r_{\text{jerk}} + r_{\text{act}} + r_{\text{force}}}_{\text{penalties}} + r_{\text{success}}$$

แต่ละ term:

| Term | สมการ |
|------|--------|
| $r_{\text{reach}}$ | $0.5 \times (1 - \tanh(5 \times d))$ โดย $d = \|\text{palm} - \text{plate}\|$ |
| $r_{\text{contact}}$ | $1.0 \times \mathbb{1}[F > 0.5]$ |
| $r_{\text{clean}}$ | $10.0 \times \Delta_{\text{clean}}$ |
| $r_{\text{sweep}}$ | $0.3 \times \|\dot{p}_{xy}\| \times \mathbb{1}[\text{contact}]$ |
| $r_{\text{time}}$ | $-0.01$ per step |
| $r_{\text{jerk}}$ | $-0.05 \times \|a_t - a_{t-1}\|^2$ |
| $r_{\text{act}}$ | $-0.005 \times \|a_t\|^2$ |
| $r_{\text{force}}$ | $-0.01 \times \max(0, F - 50)$ |
| $r_{\text{success}}$ | $+50$ one-shot เมื่อ cleaned $\geq 0.95$ |

### 📊 Artifact Interpretation

**`reward_contract.json`** — เอกสาร reward 9 terms พร้อม safety rules:
```json
{
  "version": "2.0",
  "terms": {
    "r_clean":   {"weight": 10.0, "sign": "+"},
    "r_reach":   {"weight": 0.5,  "sign": "+"},
    "r_contact": {"weight": 1.0,  "sign": "+"},
    "r_sweep":   {"weight": 0.3,  "sign": "+"},
    "r_success": {"weight": 50.0, "sign": "+"},
    "r_time":    {"weight": 0.01, "sign": "-"},
    "r_jerk":    {"weight": 0.05, "sign": "-"},
    "r_act":     {"weight": 0.005,"sign": "-"},
    "r_force":   {"weight": 0.01, "sign": "-"}
  },
  "safety": {"F_soft": 50.0, "F_hard": 200.0}
}
```
> 🔍 **แปลผล**: contract นี้เป็น "ground truth" ที่ทุก NB ถัดไปอ้างอิง — ถ้าแก้ weight ใน code แต่ไม่แก้ JSON ถือว่าผิด

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่ต้องได้ |
|-------|-------------|
| reward_contract.json | 9 terms, version "2.0" |
| Test episodes mean_reward | ≈ -0.006 (ช่วง -0.01 ถึง -0.003) |
| Safety: F_soft / F_hard | 50 N / 200 N |
| MLflow connection | เชื่อมต่อได้ (run logged) |
| Reward breakdown | time+jerk+act penalty มากกว่า reach reward |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| MLflow connection error | credentials ผิด | ตรวจ `.env.local` |
| Reward ≈ -0.006 | ปกติสำหรับ random policy | time+jerk+act penalty > reaching reward |

---

## NB05 — Baselines & Smoothing

📁 `notebooks/NB05_baselines_smoothing.ipynb`

### Objective
สร้าง baseline policies เพื่อเปรียบเทียบกับ RL:
1. **Random** — สุ่ม action
2. **Heuristic** — ชี้มือไปที่จาน (palm → plate)
3. **SmoothActionWrapper** — EMA smoothing (alpha=0.3)
4. **BaseController** — heuristic + EMA (สำหรับ NB08 Residual)
5. **Leaderboard** — ตารางเปรียบเทียบทุก baseline

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB01–NB04 ผ่าน | NB01–NB04 |
| Reward contract | NB04 |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `baseline_leaderboard.csv` | `artifacts/NB05/` | ตารางเปรียบเทียบ baselines |
| `nb05_config.json` | `artifacts/NB05/` | Config |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "eval_episodes": 20,
    "eval_steps": 200,
    "smooth_alpha": 0.3,  # EMA สำหรับ SmoothActionWrapper
}
```

### Expected Runtime
- **CPU**: 5–15 นาที (หลาย episodes)
- **GPU**: 2–5 นาที

### 📐 Algorithm: EMA Smoothing (SmoothActionWrapper)

NB05 ใช้ **Exponential Moving Average (EMA)** สำหรับ smooth action:

$$\bar{a}_t = \alpha \cdot a_t + (1 - \alpha) \cdot \bar{a}_{t-1}$$

โดย:
- $a_t$ = raw action จาก policy
- $\bar{a}_t$ = smoothed action ที่ส่งเข้า env
- $\alpha = 0.3$ (default) → action เปลี่ยนช้าๆ ลด jerk แต่ยัง responsive

**ทำไมต้อง smooth?**
- Reward มี jerk penalty: $r_{\text{jerk}} = -0.05 \times \|a_t - a_{t-1}\|^2$
- ถ้า action กระตุก (jerk สูง) agent โดนปรับหนัก
- EMA ทำให้ action ต่อเนื่อง → jerk penalty ลด → reward ดีขึ้น
- ใน robotics จริง การเคลื่อนไหวแบบ nuuนวลช่วยยืดอายุการใช้งานของ actuator

### BaseController (Heuristic + EMA)

BaseController ใช้ใน NB08 (Residual SAC) เป็น `a_base`:

```python
class BaseController:
    def predict(self, obs):
        palm_to_plate = obs["palm_to_plate"]      # ทิศทางไปจาน
        raw_action = direction_to_joint_delta(...)  # แปลงเป็น joint delta
        return ema_smooth(raw_action, alpha=0.3)    # EMA smooth
```

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่ต้องได้ |
|-------|-------------|
| Random success rate | 0% |
| Heuristic cleaned_ratio | 10–20% (200 steps) |
| BaseController jerk | ต่ำกว่า raw heuristic (เพราะ EMA) |
| Leaderboard: heuristic > random | ✅ |
| baseline_leaderboard.csv | มีครบทุก method |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| Heuristic cleaned = 0 | ยังไม่ถึงจาน 200 steps | เพิ่ม `eval_steps` |
| BaseController ไม่ smooth | alpha ต่ำ/สูงเกิน | ปรับ `smooth_alpha` (0.3 เหมาะ) |

---

## NB06 — Train PPO

📁 `notebooks/NB06_train_ppo.ipynb`

### Objective
เทรน **PPO** (Proximal Policy Optimization) ด้วย Stable-Baselines3:
- On-policy algorithm
- Vectorized env (4 envs บน GPU)
- Learning curve + evaluation

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB05 ผ่าน (baselines สร้างแล้ว) | NB05 |
| **GPU** (strongly recommended) | RunPod |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `ppo_model.zip` | `artifacts/NB06/` | Trained PPO model |
| `learning_curve.png` | `artifacts/NB06/` | Reward vs steps plot |
| `eval_results.json` | `artifacts/NB06/` | Final eval metrics |
| `train_log.csv` | `artifacts/NB06/` | Training log (every N steps) |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "total_timesteps": 500_000,  # GPU / 20_000 CPU
    "n_envs": 4,                  # GPU / 1 CPU
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 256,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "net_arch": [256, 256],
}
```

### Expected Runtime
- **CPU**: หลายชั่วโมง (ไม่แนะนำ)
- **GPU (4090)**: 1–3 ชม.

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| `CUDA out of memory` | n_envs/batch_size ใหญ่ | ลด n_envs เป็น 2 |
| Reward ไม่ขึ้น | lr สูงเกิน / ent_coef ต่ำ | ลด lr, เพิ่ม ent_coef |
| Training ช้ามาก (CPU) | ใช้ CPU | ย้ายไป RunPod GPU |

> ⚠️ **ห้ามกด Run All** บนเครื่อง CPU — จะใช้เวลานานมาก

---

## NB07 — Train SAC

📁 `notebooks/NB07_train_sac.ipynb`

### Objective
เทรน **SAC** (Soft Actor-Critic) ด้วย SB3:
- Off-policy algorithm (มี replay buffer)
- ต้องการ env เดียว (ไม่ vectorize)
- Automatic entropy tuning

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB05 ผ่าน | NB05 |
| **GPU** (strongly recommended) | RunPod |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `sac_model.zip` | `artifacts/NB07/` | Trained SAC model |
| `learning_curve.png` | `artifacts/NB07/` | Reward vs steps plot |
| `eval_results.json` | `artifacts/NB07/` | Final eval metrics |
| `train_log.csv` | `artifacts/NB07/` | Training log |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "total_timesteps": 500_000,  # GPU / 20_000 CPU
    "learning_rate": 3e-4,
    "buffer_size": 1_000_000,     # GPU / 50_000 CPU
    "batch_size": 256,
    "ent_coef": "auto",
    "tau": 0.005,
    "gamma": 0.99,
    "learning_starts": 1000,
    "net_arch": [256, 256],
}
```

### Expected Runtime
- **CPU**: หลายชั่วโมง (ไม่แนะนำ)
- **GPU (4090)**: 1–3 ชม.

### 📐 Algorithm: SAC (Soft Actor-Critic)

SAC maximize **entropy-augmented objective**:

$$J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t)} \left[ r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t)) \right]$$

โดย:
- $\mathcal{H}(\pi) = -\mathbb{E}[\log \pi(a|s)]$ — entropy (ความหลากหลายของ action distribution)
- $\alpha$ — temperature parameter (เมื่อ `ent_coef="auto"` จะ tune อัตโนมัติ)

**Critic Update** (ใช้ 2 Q-networks ลด overestimation):

$$Q_{\text{target}} = r + \gamma \left( \min_{i=1,2} Q_{\bar{\theta}_i}(s', a') - \alpha \log \pi(a'|s') \right)$$

**Soft Target Update** (อัปเดต target ช้าๆ):

$$\bar{\theta} \leftarrow \tau \theta + (1 - \tau) \bar{\theta}, \quad \tau = 0.005$$

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Phase | Steps | สิ่งที่คาดหวัง |
|-------|-------|------------------|
| Warmup (0–1K) | reward ไม่ขึ้น | เก็บ data ก่อน (`learning_starts=1000`) |
| Early (1K–100K) | reward ขึ้นเร็ว | replay buffer เริ่มเต็ม → sample efficient |
| Mid (100K–300K) | clean และ contact เพิ่ม | เรียนรู้ wiping strategy |
| Late (300K–500K) | reward plateau สูงกว่า PPO | cleaned 80–95% |
| sac_model.zip | loadable | `SAC.load()` ได้ |
| success_rate (eval) | 60–85% | ดีกว่า PPO (sample efficient) |

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| `CUDA out of memory` | buffer_size ใหญ่ | ลด buffer_size |
| Reward ต่ำช่วงแรก | learning_starts ยังไม่ครบ | ปกติ — รอให้ผ่าน learning_starts |
| NaN loss | lr สูงเกิน | ลด lr เป็น 1e-4 |

---

## NB08 — Residual SAC + Ablation

📁 `notebooks/NB08_residual_sac_ablation.ipynb`

### Objective
เทรน **Residual SAC** — ใช้ BaseController (จาก NB05) เป็นฐาน แล้วให้ SAC เรียนรู้ "ส่วนเพิ่ม" (residual):

```
a_final = clip(a_base + β × a_residual, -1, 1)
```

- `a_base` = action จาก BaseController (heuristic + EMA)
- `a_residual` = action จาก SAC policy
- `β` = scaling factor (ablation: 0.25, 0.5, 1.0)

### 📐 Algorithm: Residual Policy Learning

$$a_{\text{final}} = \text{clip}(a_{\text{base}} + \beta \times a_{\text{residual}}, -1, +1)$$

โดย:
- $a_{\text{base}}$ = BaseController output (heuristic + EMA, จาก NB05)
- $a_{\text{residual}}$ = SAC policy output (ส่วนที่เรียนรู้)
- $\beta \in \{0.25, 0.5, 1.0\}$ = scaling factor

### ทำไมต้องใช้ Residual?

| \#  | เหตุผล |
|-----|--------|
| Informed exploration | SAC สำรวจรอบๆ base action แทนที่จะสุ่มทั้ง action space |
| เรียนเร็ว | ไม่ต้องเรียน "reaching" จากศูนย์ |
| Safety net | ถ้า SAC ยังเรียนไม่ดี ก็ยังมี base พยุง |

### β Ablation Design

| β | ความหมาย | Hypothesis |
|---|---------|------------|
| 0.25 | SAC มีอิทธิพลน้อย — ส่วนใหญ่ใช้ base | เสถียรแต่ clean ได้น้อย |
| 0.5 | balance ระหว่าง base กับ SAC | **น่าจะดีที่สุด** (sweet spot) |
| 1.0 | SAC มีอิทธิพลเท่า base | อาจ overfit / ไม่เสถียร |

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่คาดหวัง |
|-------|------------------|
| β=0.5 reward | สูงกว่า standalone SAC |
| β=0.5 success_rate | 70–90% |
| ablation_plot.png | เห็นความต่างระหว่าง 3 β |
| Early learning speed | เร็วกว่า SAC from scratch |
| Training converges | ไม่ diverge / NaN |

### ทำไมต้องใช้ Residual?
- BaseController มีพื้นฐานดี (เอื้อมถึงจาน) แต่ไม่ฉลาดพอจะล้างครบ
- SAC จาก 0 ต้องเรียนรู้ทุกอย่าง (เข้าใกล้ + สัมผัส + เช็ด)
- Residual: SAC แค่เรียนรู้ส่วนที่ BaseController ทำไม่ได้ → เรียนเร็วกว่า

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| NB05 ผ่าน (BaseController) | NB05 |
| SAC hyperparams | NB07 config |
| **GPU** | RunPod |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `residual_sac_beta0.25.zip` | `artifacts/NB08/` | Model β=0.25 |
| `residual_sac_beta0.5.zip` | `artifacts/NB08/` | Model β=0.5 |
| `residual_sac_beta1.0.zip` | `artifacts/NB08/` | Model β=1.0 |
| `ablation_beta_table.csv` | `artifacts/NB08/` | ตารางเปรียบเทียบ β |
| `ablation_plot.png` | `artifacts/NB08/` | Plot เปรียบเทียบ β |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "total_timesteps": 500_000,
    "betas": [0.25, 0.5, 1.0],
    "sac_lr": 3e-4,
    "buffer_size": 1_000_000,
    "batch_size": 256,
    "smooth_alpha": 0.3,
}
```

### Expected Runtime
- **GPU (4090)**: 3–6 ชม. (3 runs × 1–2 ชม.)

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| β=1.0 ชนะทุก β | Budget ไม่พอ | เพิ่ม total_timesteps |
| BaseController ทำงานผิด | import ผิด version | ตรวจว่า import จาก NB05 ถูกต้อง |

---

## NB09 — Evaluation & Summary

📁 `notebooks/NB09_evaluation.ipynb`

### Objective
ประเมินผลทุก method อย่างเป็นระบบ:
1. Load models จาก NB06/07/08
2. Run deterministic evaluation (100 episodes per method)
3. คำนวณ **Bootstrap 95% CI** สำหรับทุก metric
4. สร้าง comparison plots
5. สร้าง video ของ best method (ถ้า render ได้)
6. สรุปผล

### Required Inputs
| สิ่งที่ต้องมี | ที่มา |
|-------------|------|
| `ppo_model.zip` | NB06 |
| `sac_model.zip` | NB07 |
| `residual_sac_beta*.zip` (best β) | NB08 |
| Baseline functions | NB05 |

### Produced Outputs / Artifacts
| ไฟล์ | Path | คำอธิบาย |
|------|------|---------|
| `eval_table.csv` | `artifacts/NB09/` | ผลรวมทุก method (mean ± CI) |
| `eval_comparison.png` | `artifacts/NB09/` | Bar chart เปรียบเทียบ |
| `ci_table.csv` | `artifacts/NB09/` | Bootstrap CI details |
| `best_method_video.mp4` | `artifacts/NB09/` | Video ของ method ดีที่สุด |
| `final_summary.json` | `artifacts/NB09/` | สรุปผลทั้งหมด |

### Metrics ที่ประเมิน
| Metric | คำอธิบาย | ดียิ่งขึ้นเมื่อ |
|--------|---------|-------------|
| `success_rate` | สัดส่วน episode ที่ล้างครบ 95% | สูง ↑ |
| `final_cleanliness` | cleaned_ratio ตอนจบ episode | สูง ↑ |
| `steps_to_95` | จำนวน step ที่ใช้ถึง 95% | ต่ำ ↓ |
| `mean_reward` | reward เฉลี่ยต่อ episode | สูง ↑ |
| `mean_jerk` | ‖aₜ − aₜ₋₁‖² เฉลี่ย | ต่ำ ↓ |
| `p95_jerk` | percentile 95 ของ jerk | ต่ำ ↓ |
| `mean_contact_force` | แรงเฉลี่ย (N) | ปานกลาง |
| `safety_violation_rate` | สัดส่วน episode ที่ force > hard limit | ต่ำ ↓ |

### 📐 Algorithm: Bootstrap Confidence Interval

NB09 ใช้ **percentile bootstrap** สำหรับ 95% CI:

1. เก็บ data = $[x_1, x_2, \ldots, x_n]$ จาก $n$ episodes
2. สุ่ม resample (with replacement) ขนาด $n$ → คำนวณ $\bar{x}^*$
3. ทำซ้ำ $B = 1000$ ครั้ง → ได้ $[\bar{x}^*_1, \ldots, \bar{x}^*_B]$
4. เรียงลำดับ → $\text{CI}_{95\%} = [\bar{x}^*_{(25)}, \bar{x}^*_{(975)}]$

$$\text{CI}_{1-\alpha} = \left[ \hat{\theta}^*_{\alpha/2}, \; \hat{\theta}^*_{1-\alpha/2} \right]$$

**ทำไมใช้ Bootstrap แทน Normal CI?**
- ไม่ต้องสมมติ distribution (non-parametric)
- ทำงานได้ดีกับ metrics ที่ไม่ได้เป็น normal เช่น success_rate (0 หรือ 1)
- ใช้ง่าย ไม่ต้อง derive สูตร variance

### Jerk Metric (ความ smooth)

$$\text{jerk}_t = \|a_t - a_{t-1}\|^2 = \sum_{d=1}^{25} (a_{t,d} - a_{t-1,d})^2$$

$$\text{mean\_jerk} = \frac{1}{T} \sum_{t=1}^{T} \text{jerk}_t$$

ค่า jerk ต่ำ = การเคลื่อนไหว smooth (ดี), jerk สูง = กระตุก (ไม่ดี)

### ✅ สิ่งที่คาดหวัง (Expected Outcomes)

| Check | ค่าที่คาดหวัง |
|-------|------------------|
| eval_table.csv | มีทุก method (random, heuristic, ppo, sac, residual_sac) |
| Residual SAC > SAC > PPO | success rate (hypothesis) |
| Random success_rate | 0% |
| CI width (100 eps) | ≤ ±10% สำหรับ success_rate |
| safety_violation_rate | < 2% สำหรับ trained models |
| eval_comparison.png | bar chart + error bars ทุก metric |

### Key Config Knobs
```python
CFG = {
    "seed": 42,
    "eval_episodes": 100,
    "bootstrap_samples": 1000,
    "confidence_level": 0.95,
    "deterministic": True,
}
```

### Expected Runtime
- **CPU**: 15–30 นาที
- **GPU**: 5–10 นาที

### Common Pitfalls
| ปัญหา | สาเหตุ | วิธีแก้ |
|--------|--------|--------|
| `FileNotFoundError: ppo_model.zip` | ยังไม่เทรน | รัน NB06 ก่อน |
| Video render fails | ไม่มี GPU | ใช้ fallback (สร้าง plot แทน video) |
| CI กว้างมาก | episodes น้อย | เพิ่ม eval_episodes |

---

## สรุปตาราง I/O ทุก NB

| NB | Input Files | Output Artifacts | Key Metric |
|----|-------------|------------------|------------|
| 01 | `src/envs/`, `.env.local` | `env_spec.json`, `active_joints.json` | obs=(1,168), act=(25,) |
| 02 | NB01 env | `grid_trace.csv`, `grid_*.png` | 100/100 cells mapped |
| 03 | NB01 env | `brush_effect_demo.png`, `cleaning_trace.csv` | radius=1 → 9 cells/touch |
| 04 | NB01 env, `.env.local` | `reward_contract.json` | reward ≈ -0.006 (random) |
| 05 | NB01–04 | `baseline_leaderboard.csv` | heuristic > random |
| 06 | NB05 | `ppo_model.zip`, `learning_curve.png` | reward ↑, cleaned ↑ |
| 07 | NB05 | `sac_model.zip`, `learning_curve.png` | reward ↑, cleaned ↑ |
| 08 | NB05, NB07 config | `residual_sac_beta*.zip`, `ablation_plot.png` | best β selection |
| 09 | NB06/07/08 models | `eval_table.csv`, `eval_comparison.png` | method comparison |

---

*ก่อนหน้า → [03 — Environment & Task](03_environment_and_task.md) | ต่อไป → [05 — RL Methods Tutorial](05_rl_methods_tutorial.md)*
