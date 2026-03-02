# 06 — Experiment Tracking (MLflow + CSV)

> เอกสารนี้อธิบายวิธี track experiments ด้วย MLflow และทางเลือก CSV logger

---

## สารบัญ

- [ทำไมต้อง Track Experiments?](#ทำไมต้อง-track-experiments)
- [MLflow คืออะไร](#mlflow-คืออะไร)
- [โครงสร้าง Experiment](#โครงสร้าง-experiment)
- [สิ่งที่ควร Log](#สิ่งที่ควร-log)
- [การตั้งค่า MLflow](#การตั้งค่า-mlflow)
- [วิธีใช้ MLflow ใน Notebook](#วิธีใช้-mlflow-ใน-notebook)
- [Naming Conventions](#naming-conventions)
- [ทางเลือก: CSV Logger](#ทางเลือก-csv-logger)
- [ข้อควรระวัง](#ข้อควรระวัง)

---

## ทำไมต้อง Track Experiments?

เมื่อเทรน RL หลาย algorithm, หลาย seed, หลาย hyperparameter — ถ้าไม่บันทึกอย่างเป็นระบบ จะ:
- **ลืม** ว่ารัน parameter อะไร
- **เปรียบเทียบไม่ได้** — ไม่รู้ว่า run ไหนดีกว่า
- **ทำซ้ำไม่ได้** — reproduce ผลไม่ได้

### สิ่งที่ต้องบันทึก

| ประเภท | ตัวอย่าง |
|--------|---------|
| **Parameters** | learning_rate, batch_size, seed, n_envs, beta |
| **Metrics** | mean_reward, success_rate, cleaned_ratio, loss |
| **Artifacts** | model.zip, learning_curve.png, eval_results.json |
| **Tags** | algorithm="PPO", stage="training", nb="NB06" |

---

## MLflow คืออะไร

[MLflow](https://mlflow.org/) เป็น open-source platform สำหรับ manage ML experiments ประกอบด้วย:

1. **Tracking**: บันทึก parameters, metrics, artifacts
2. **UI**: เว็บ dashboard ดู/เปรียบเทียบ runs
3. **Model Registry**: จัดการ model versions

โปรเจกต์นี้ใช้ MLflow Tracking Server ที่: `https://mlflow.cie.co.th`

---

## โครงสร้าง Experiment

```
Experiment: "g1_fullbody_apple_dishwipe"
│
├── Run: NB01_setup_smoke_v2
│   ├── params: seed=42, env_id=UnitreeG1PlaceAppleInBowlFullBody-v1, ...
│   ├── metrics: apple_obs_dim=~110, act_dim=37
│   └── artifacts: env_spec.json, active_joints.json
│
├── Run: NB03_reward_contract_v2
│   ├── params: seed=42, test_episodes=5
│   ├── metrics: reward_range=[min, max]
│   └── artifacts: reward_contract.json
│
|├── Run: NB05_ppo_seed42_2M
│   ├── params: lr=3e-4→1e-5, batch_size=2048, n_envs=64, net=[512,512], ...
│   ├── metrics: final_reward=XX, success_rate=XX
│   └── artifacts: ppo_apple.zip, learning_curve.png
│
├── Run: NB06_sac_seed42_2M
│   └── ...
│
├── Run: NB07_residual_beta0.5_seed42
│   └── ...
│
├── Run: NB08_evaluation_200ep
│   └── artifacts: best_method.json, comparison_plot.png
│
└── Run: NB09_bonus_dishwipe_winner
    └── artifacts: {winner}_dishwipe.zip, cross_task_comparison.png
```

---

## สิ่งที่ควร Log

### Per Notebook

| NB | Params | Metrics | Artifacts |
|----|--------|---------|-----------|
| NB01 | seed, env_id, obs_mode | obs_dim, act_dim | env_spec.json |
| NB02 | seed, n_steps | reward_per_step, obs_mapping | env_exploration_trace.csv |
| NB03 | seed, test_eps | reward_range, fall_detection | reward_contract.json |
| NB04 | seed, eval_episodes | random_reward, heuristic_reward | baseline_leaderboard.csv |
| NB05 | lr, batch, n_envs, gamma, clip, net_arch | final_reward, success_rate | ppo_apple.zip, learning_curve.png |
| NB06 | lr, batch, buffer, tau, gamma, net_arch | final_reward, success_rate | sac_apple.zip, learning_curve.png |
| NB07 | betas, lr, buffer, net_arch | best_beta, best_reward | residual_apple_beta*.zip, ablation_plot.png |
| NB08 | eval_eps, bootstrap_n, stat_tests | per-method metrics | best_method.json, comparison.png |
| NB09 | winner_method, total_steps | dishwipe_reward, cross_task | {winner}_dishwipe.zip, cross_task.png |

---

## การตั้งค่า MLflow

### 1. สร้าง `.env.local`

```bash
cp .env.example .env.local
```

แก้ไข `.env.local`:
```dotenv
MLFLOW_TRACKING_URI=https://mlflow.cie.co.th
MLFLOW_TRACKING_USERNAME=your_username
MLFLOW_TRACKING_PASSWORD=your_password
```

> ⚠️ **ห้ามใส่ credentials จริงลง `.env.example`** — ไฟล์นั้น commit ขึ้น Git  
> ⚠️ `.env.local` อยู่ใน `.gitignore` แล้ว

### 2. ติดตั้ง mlflow (อยู่ใน requirements แล้ว)
```bash
pip install mlflow
```

### 3. ทดสอบการเชื่อมต่อ
```python
import os
import mlflow

os.environ["MLFLOW_TRACKING_URI"] = "https://mlflow.cie.co.th"
os.environ["MLFLOW_TRACKING_USERNAME"] = "your_username"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your_password"

mlflow.set_experiment("dishwipe_unitree_g1")
print("✅ MLflow connected")
```

---

## วิธีใช้ MLflow ใน Notebook

### Pattern มาตรฐาน

```python
import mlflow
import os
from pathlib import Path

# โหลด credentials จาก .env.local
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env.local")

tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
if tracking_uri:
    os.environ["MLFLOW_TRACKING_URI"] = tracking_uri
    os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("MLFLOW_TRACKING_USERNAME", "")
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("MLFLOW_TRACKING_PASSWORD", "")
    mlflow.set_experiment("dishwipe_unitree_g1")

# --- เริ่ม run ---
with mlflow.start_run(run_name="NB06_ppo_seed42_500k"):
    # Log parameters
    mlflow.log_params({
        "seed": 42,
        "algorithm": "PPO",
        "learning_rate": "3e-4->1e-5",
        "total_timesteps": 2_000_000,
        "n_envs": 64,
        "net_arch": "[512,512]",
        "activation": "ReLU",
    })

    # ... training ...

    # Log metrics
    mlflow.log_metrics({
        "final_reward": 25.3,
        "success_rate": 0.65,
        "cleaned_ratio": 0.89,
    })

    # Log artifacts (files)
    mlflow.log_artifact("artifacts/NB06/ppo_model.zip")
    mlflow.log_artifact("artifacts/NB06/learning_curve.png")
```

### NB04 สร้าง Helper Functions

NB04 สร้าง helper functions `setup_mlflow()` และ `log_training_run()` ที่ NB ถัดไปใช้ได้:

```python
def setup_mlflow(experiment_name="dishwipe_unitree_g1"):
    """ตั้งค่า MLflow จาก .env.local — ถ้าไม่มี credentials จะ skip"""
    ...

def log_training_run(run_name, params, metrics, artifact_paths):
    """Log training run อย่างครบถ้วน"""
    ...
```

---

## Naming Conventions

### Run Names

ใช้ pattern: `{NB}_{algorithm}_{key_params}`

| NB | Run Name ตัวอย่าง |
|----|--------------------|
| NB01 | `NB01_setup_smoke_v2` |
| NB02 | `NB02_env_exploration_v1` |
| NB03 | `NB03_reward_contract_v1` |
| NB04 | `NB04_baselines_v1` |
| NB05 | `NB05_ppo_seed42_2M` |
| NB06 | `NB06_sac_seed42_2M` |
| NB07 | `NB07_residual_beta0.5_seed42` |
| NB08 | `NB08_eval_200ep_v1` |
| NB09 | `NB09_bonus_dishwipe_winner` |

### Metric Names

ใช้ชื่อที่สอดคล้องกันทั้งโปรเจกต์:

| Metric | คำอธิบาย |
|--------|---------|
| `cleaned_ratio` | สัดส่วนที่ล้างแล้ว (0-1) |
| `success_rate` | สัดส่วน episode ที่สำเร็จ (0-1) |
| `mean_reward` | reward เฉลี่ยต่อ episode |
| `steps_to_95` | จำนวน steps ถึง 95% clean |
| `mean_jerk` | ‖aₜ − aₜ₋₁‖² เฉลี่ย |
| `p95_jerk` | percentile 95 ของ jerk |
| `fz_mean` | แรงสัมผัสเฉลี่ย (N) |
| `fz_p95` | percentile 95 ของแรง (N) |
| `safety_violation_rate` | สัดส่วน episode ที่เกิน force limit |

---

## ทางเลือก: CSV Logger

ถ้า MLflow ใช้ไม่ได้ (เช่น ไม่มี server, offline) สามารถใช้ CSV logger แทน:

### วิธีใช้

```python
import csv
from pathlib import Path

def log_to_csv(filepath: str, data: dict):
    """Append one row to CSV — create file if not exists"""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)

# ตัวอย่างการใช้
log_to_csv("artifacts/NB06/train_log.csv", {
    "step": 10000,
    "mean_reward": -0.003,
    "cleaned_ratio": 0.15,
    "loss": 0.045,
})
```

### โครงสร้าง artifacts/

```
artifacts/
├── NB01/
│   ├── env_spec.json
│   ├── active_joints.json
│   └── nb01_config.json
├── NB06/
│   ├── ppo_model.zip
│   ├── learning_curve.png
│   ├── train_log.csv        ← CSV logger
│   └── eval_results.json
└── NB09/
    ├── eval_table.csv
    └── eval_comparison.png
```

---

## ข้อควรระวัง

### 1. ห้าม Hardcode Secrets

❌ ผิด:
```python
mlflow.set_tracking_uri("https://mlflow.cie.co.th")
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "secret123"
```

✅ ถูก:
```python
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env.local")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
```

### 2. Log Parameters ก่อน Training

ถ้า training crash กลางคัน อย่างน้อยยังมี parameters บันทึกไว้

### 3. Log Artifacts หลัง Training

```python
with mlflow.start_run():
    mlflow.log_params(cfg)     # log ก่อน
    model.learn(...)           # training
    mlflow.log_metrics(...)    # log ผล
    mlflow.log_artifact(...)   # save artifacts
```

### 4. ไม่ต้อง log ทุก step

- Training metrics: log ทุก 10,000–50,000 steps
- Eval metrics: log ตอนจบ training
- Model: save ตอนจบ (+ checkpoints ทุก 200K steps)

### 5. MLflow ล่มไม่กระทบ Training

Notebook ทุกอันมี try/except สำหรับ MLflow — ถ้า server ล่ม training ยังรันต่อได้ artifacts ยังอยู่ใน disk

---

*ก่อนหน้า → [05 — RL Methods Tutorial](05_rl_methods_tutorial.md) | ต่อไป → [07 — Evaluation & Reporting](07_evaluation_and_reporting.md)*
