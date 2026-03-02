# 01 — Setup บนเครื่อง Local (Windows / macOS / Linux)

> คู่มือนี้สอนวิธีตั้งค่าโปรเจกต์ตั้งแต่ 0 บนเครื่องส่วนตัว (ไม่ต้องมี GPU)

---

## สารบัญ

- [ก่อนเริ่ม — สิ่งที่ต้องมี](#ก่อนเริ่ม--สิ่งที่ต้องมี)
- [ขั้นตอนที่ 1 — Clone Repository](#ขั้นตอนที่-1--clone-repository)
- [ขั้นตอนที่ 2 — สร้าง Virtual Environment](#ขั้นตอนที่-2--สร้าง-virtual-environment)
- [ขั้นตอนที่ 3 — ติดตั้ง Dependencies](#ขั้นตอนที่-3--ติดตั้ง-dependencies)
- [ขั้นตอนที่ 4 — ตั้งค่า Environment Variables (.env.local)](#ขั้นตอนที่-4--ตั้งค่า-environment-variables-envlocal)
- [ขั้นตอนที่ 5 — รัน Notebook ใน VS Code](#ขั้นตอนที่-5--รัน-notebook-ใน-vs-code)
- [ขั้นตอนที่ 6 — Smoke Test (NB01)](#ขั้นตอนที่-6--smoke-test-nb01)
- [.gitignore — สิ่งที่ห้าม commit](#gitignore--สิ่งที่ห้าม-commit)
- [Git Workflow แนะนำ](#git-workflow-แนะนำ)
- [Troubleshooting](#troubleshooting)

---

## ก่อนเริ่ม — สิ่งที่ต้องมี

| สิ่งที่ต้องใช้ | เวอร์ชันแนะนำ | วิธีตรวจ |
|---------------|--------------|---------|
| **Python** | 3.11 – 3.14 | `python --version` |
| **pip** | 23+ | `pip --version` |
| **Git** | 2.40+ | `git --version` |
| **VS Code** | ล่าสุด | เปิดแอป |
| VS Code Extension: **Jupyter** | ล่าสุด | Install จาก Extensions Panel |
| VS Code Extension: **Python** | ล่าสุด | Install จาก Extensions Panel |

> **หมายเหตุ**: Python 3.14 ทำงานได้ แต่ถ้ามีปัญหา compatibility ให้ใช้ 3.11 หรือ 3.12

---

## ขั้นตอนที่ 1 — Clone Repository

### Windows (PowerShell)
```powershell
cd C:\Users\<ชื่อคุณ>\Desktop
git clone https://github.com/siriponsri/robotic-sim-dishwash.git
cd robotic-sim-dishwash
```

### macOS / Linux (Terminal)
```bash
cd ~/Desktop
git clone https://github.com/siriponsri/robotic-sim-dishwash.git
cd robotic-sim-dishwash
```

> **ข้อควรระวัง**: ถ้า path มีช่องว่าง (เช่น `My Project`) อาจทำให้บาง tool มีปัญหา แนะนำ path ไม่มีช่องว่าง

---

## ขั้นตอนที่ 2 — สร้าง Virtual Environment

โปรเจกต์นี้ใช้โฟลเดอร์ `.env/` เป็น virtual environment (ไม่ใช่ไฟล์ `.env` สำหรับ env vars)

### Windows (PowerShell)
```powershell
python -m venv .env
.env\Scripts\Activate.ps1
```

> ⚠️ ถ้าเจอ error เรื่อง **Execution Policy** ให้รันคำสั่งนี้ก่อน (ในฐานะ Admin):
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### macOS / Linux
```bash
python3 -m venv .env
source .env/bin/activate
```

### ตรวจว่า activate สำเร็จ
```
# ควรเห็น (.env) หน้า prompt
(.env) PS C:\...\robotic-sim-dishwash>
```

---

## ขั้นตอนที่ 3 — ติดตั้ง Dependencies

```bash
# อัปเกรด pip ก่อน
python -m pip install --upgrade pip

# ติดตั้ง dependencies
python -m pip install -r requirements.runpod.txt
```

> **ทำไมใช้ `requirements.runpod.txt`?** — เพราะไฟล์นี้มี dependencies ครบทั้ง simulation, RL training และ notebook stack ส่วน `ref-code/requirements.txt` มีแค่ส่วน simulation

### ตรวจว่าติดตั้งสำเร็จ
```bash
python scripts/runpod_verify.py
```

ผลที่ต้องได้:
```
[OK] All core imports succeeded
[OK] Torch version: 2.10.0+cpu
[OK] CUDA available: False
```

> `CUDA available: False` บนเครื่อง local ที่ไม่มี GPU ถือว่าปกติ

---

## ขั้นตอนที่ 4 — ตั้งค่า Environment Variables (.env.local)

โปรเจกต์ใช้ MLflow สำหรับ experiment tracking ต้องตั้งค่า credentials:

### 4.1 Copy template
```bash
# ไฟล์ .env.example มีให้แล้วใน repo
# สร้างไฟล์ .env.local จาก template
```

### Windows (PowerShell)
```powershell
Copy-Item .env.example .env.local
```

### macOS / Linux
```bash
cp .env.example .env.local
```

### 4.2 แก้ไข `.env.local`

เปิดไฟล์ `.env.local` แล้วใส่ค่าจริง:

```dotenv
# ── MLflow ────────────────────────────────────────────────
MLFLOW_TRACKING_URI=https://mlflow.example.com
MLFLOW_TRACKING_USERNAME=your_actual_username
MLFLOW_TRACKING_PASSWORD=your_actual_password
```

> ⚠️ **ห้ามใส่ secrets จริงลงใน `.env.example`** — ไฟล์นั้น commit ขึ้น Git ได้  
> ⚠️ **ไฟล์ `.env.local` อยู่ใน `.gitignore` แล้ว** จะไม่ถูก commit

### 4.3 วิธีโหลดใน Notebook

ทุก notebook จะโหลด credentials แบบนี้:
```python
import os
from pathlib import Path
from dotenv import load_dotenv  # pip install python-dotenv (ถ้ายังไม่มี)

load_dotenv(Path(PROJECT_ROOT) / ".env.local")
tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "")
```

---

## ขั้นตอนที่ 5 — รัน Notebook ใน VS Code

### 5.1 เปิดโปรเจกต์
```
File → Open Folder → เลือกโฟลเดอร์ robotic-sim-dishwash
```

### 5.2 เลือก Python Interpreter
1. กด `Ctrl+Shift+P` (หรือ `Cmd+Shift+P` บน Mac)
2. พิมพ์ `Python: Select Interpreter`
3. เลือก `.env` ที่อยู่ใน project (เช่น `.\.env\Scripts\python.exe`)

### 5.3 เปิด Notebook
1. ไปที่ `notebooks/NB01_setup_smoke.ipynb`
2. VS Code จะเปิดใน Notebook Editor
3. ตรวจ kernel ขวาบน — ต้องเป็น `.env (Python 3.x.x)`

### 5.4 วิธีรัน Cell
- **รัน cell เดียว**: กด `Shift+Enter` หรือกดปุ่ม ▶️ ข้าง cell
- **รัน ทั้ง notebook**: กด `Run All` ด้านบน (แต่ระวัง! ดูหัวข้อถัดไป)

### ⚠️ การรัน Notebook แบบไม่เผลอเทรน

NB06–NB08 มี cell ที่ **เทรน model** ซึ่งใช้เวลานานและต้องใช้ GPU  
**ห้ามกด Run All** บน notebook เหล่านั้นถ้ายังไม่พร้อม!

วิธีปลอดภัย:
1. รัน cell **ทีละอัน** (Shift+Enter)
2. หยุดก่อนถึง cell ที่เขียนว่า `## Training Loop` หรือ `model.learn()`
3. อ่าน Markdown ก่อน cell นั้นเสมอ — จะบอกว่าใช้เวลานานแค่ไหน

---

## ขั้นตอนที่ 6 — Smoke Test (NB01)

1. เปิด `notebooks/NB01_setup_smoke.ipynb`
2. กด **Restart Kernel** (🔄) เพื่อเริ่มใหม่
3. กด **Run All**
4. ตรวจผลลัพธ์:

| สิ่งที่ต้องเห็น | ค่าที่คาดหวัง |
|----------------|--------------|
| Python version | 3.11+ |
| Apple obs dim | ~110+ |
| DishWipe obs dim | ~200+ |
| act dim | 37 (full body) |
| Robot DOF | 37 |
| Smoke test | `PASSED` |

> **Render จะ fail** บนเครื่องที่ไม่มี GPU/Vulkan — นี่คือพฤติกรรมปกติ ไม่ต้องกังวล

---

## .gitignore — สิ่งที่ห้าม commit

ไฟล์ `.gitignore` กำหนดไว้แล้ว:

```gitignore
# Secrets — ห้าม commit!
.env.local

# Virtual environment
.env/
.venv/

# Artifacts ขนาดใหญ่
artifacts/**/*.mp4

# Python cache
__pycache__/
*.pyc
```

### สิ่งที่ห้าม commit ลง Git เด็ดขาด
1. **`.env.local`** — มี password/token
2. **`.env/`** — virtual environment (ใหญ่มาก)
3. **`*.mp4`** — video files (ใหญ่)
4. **model files ขนาดใหญ่** — ถ้า `model.zip` > 100MB ให้ใช้ Git LFS หรือเก็บบน cloud

---

## Git Workflow แนะนำ

### ขั้นตอนการทำงาน

```bash
# 1. สร้าง branch ใหม่จาก main
git checkout main
git pull origin main
git checkout -b feature/my-notebook-fix

# 2. ทำงาน แก้โค้ด รัน notebook
# ...

# 3. ตรวจสถานะ
git status

# 4. เพิ่มไฟล์ที่ต้องการ commit (เลือกเฉพาะไฟล์ ไม่ใช่ git add .)
git add notebooks/NB01_setup_smoke.ipynb
git add src/envs/dishwipe_env.py

# 5. Commit
git commit -m "fix: update NB01 obs shape assertion"

# 6. Push
git push origin feature/my-notebook-fix

# 7. สร้าง Pull Request บน GitHub
```

### ข้อควรระวัง
- **ตรวจ `git diff` ก่อน commit** เสมอ — Jupyter notebook มี output ใหญ่
- แนะนำ **Clear All Outputs** ก่อน commit notebook
- ใช้ **branch** ไม่ commit ตรงบน main

---

## Troubleshooting

### 1. PowerShell: `Activate.ps1 cannot be loaded` (Windows)

**สาเหตุ**: Execution Policy ไม่อนุญาต  
**แก้**:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. `grep` ไม่มี (Windows)

Windows ไม่มี `grep` ให้ใช้ `Select-String` แทน:
```powershell
# แทน: grep -r "UnitreeG1DishWipe" src/
Get-ChildItem -Recurse src/ | Select-String "UnitreeG1DishWipe"
```

### 3. `ModuleNotFoundError: No module named 'src'`

**สาเหตุ**: Python ไม่เห็น project root ใน path  
**แก้**: ทุก notebook มีการเพิ่ม path อยู่แล้ว:
```python
import sys
from pathlib import Path
PROJECT_ROOT = str(Path.cwd().parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
```

### 4. `pkg_resources is deprecated` warning

**สาเหตุ**: SAPIEN ใช้ `pkg_resources` ซึ่ง deprecated ใน setuptools ใหม่  
**แก้**: ไม่ต้องทำอะไร — เป็นแค่ warning ไม่กระทบการทำงาน  
**ป้องกัน**: ใช้ `setuptools<81` (อยู่ใน requirements แล้ว)

### 5. `Vulkan` / `ErrorOutOfPoolMemory` / Render failed

**สาเหตุ**: ไม่มี GPU หรือ Vulkan driver  
**แก้**: ปกติสำหรับเครื่อง CPU — NB01 มี try/except จัดการแล้ว  
**หมายเหตุ**: Render จะทำงานบน RunPod ที่มี GPU

### 6. `torch.cuda.is_available()` returns `False`

**สาเหตุ**: ติดตั้ง PyTorch แบบ CPU-only  
**แก้**: ปกติสำหรับ local dev — NB01–NB05 ทำงานบน CPU ได้  
สำหรับ GPU ให้ setup บน RunPod (ดู [02_runpod_setup.md](02_runpod_setup.md))

### 7. `ipykernel` ไม่เจอ / Kernel ไม่ขึ้นใน VS Code

**แก้**:
```bash
python -m pip install ipykernel
python -m ipykernel install --user --name robotic-sim-dishwash --display-name "Python (robotic-sim-dishwash)"
```
แล้วใน VS Code กด `Select Kernel` → เลือก `Python (robotic-sim-dishwash)`

### 8. Notebook output ใหญ่ เปิดช้า

**แก้**: ก่อน commit/push ให้ Clear outputs:
- VS Code: `...` menu → `Clear All Outputs`
- Terminal: `jupyter nbconvert --clear-output --inplace notebooks/NB01_setup_smoke.ipynb`

---

*ต่อไป → [02 — Setup บน RunPod GPU](02_runpod_setup.md)*
