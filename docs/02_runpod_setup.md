# 02 — Setup บน RunPod GPU

> คู่มือนี้สอนวิธีสร้าง GPU Pod บน RunPod, เชื่อมต่อด้วย SSH, และรันโปรเจกต์แบบ step-by-step

---

## สารบัญ

- [ทำไมต้องใช้ RunPod?](#ทำไมต้องใช้-runpod)
- [สเปคขั้นต่ำที่แนะนำ](#สเปคขั้นต่ำที่แนะนำ)
- [ขั้นตอนที่ 1 — สมัครบัญชี RunPod](#ขั้นตอนที่-1--สมัครบัญชี-runpod)
- [ขั้นตอนที่ 2 — สร้าง SSH Key](#ขั้นตอนที่-2--สร้าง-ssh-key)
- [ขั้นตอนที่ 3 — เพิ่ม Public Key ลง RunPod](#ขั้นตอนที่-3--เพิ่ม-public-key-ลง-runpod)
- [ขั้นตอนที่ 4 — สร้าง Pod](#ขั้นตอนที่-4--สร้าง-pod)
- [ขั้นตอนที่ 5 — SSH เข้า Pod](#ขั้นตอนที่-5--ssh-เข้า-pod)
- [ขั้นตอนที่ 6 — VS Code Remote-SSH](#ขั้นตอนที่-6--vs-code-remote-ssh)
- [ขั้นตอนที่ 7 — Clone & Setup Project](#ขั้นตอนที่-7--clone--setup-project)
- [ขั้นตอนที่ 8 — ตรวจสอบ GPU](#ขั้นตอนที่-8--ตรวจสอบ-gpu)
- [ขั้นตอนที่ 9 — รัน Notebook](#ขั้นตอนที่-9--รัน-notebook)
- [การ Sync Code กับ GitHub](#การ-sync-code-กับ-github)
- [Troubleshooting](#troubleshooting)

---

## ทำไมต้องใช้ RunPod?

NB05–NB07 (Training PPO/SAC/Residual) + NB09 (Bonus DishWipe) ต้องใช้ **GPU** เพื่อ:
- เปิด simulation หลาย env พร้อมกัน (vectorized env)
- คำนวณ neural network ได้เร็ว
- เทรน 2M steps ภายในเวลาที่เหมาะสม (2–4 ชม. แทน 1–2 วัน บน CPU)

**RunPod** คือ cloud GPU platform ที่จ่ายตามชั่วโมง ราคาเริ่มต้น ~$0.40/ชม.

---

## สเปคขั้นต่ำที่แนะนำ

| Component | ขั้นต่ำ | แนะนำ | หมายเหตุ |
|-----------|---------|-------|---------|
| **GPU** | RTX 4090 (24 GB) | **RTX 5090** (32 GB) | เร็วกว่า ~2x, VRAM มากพอสำหรับ 64 envs |
| **VRAM** | 24 GB | 32 GB | Replay buffer 10M + model weights + 64 envs |
| **CPU Cores** | 8 | 16+ | สำหรับ vectorized env |
| **RAM** | 32 GB | 40 GB+ | Replay buffer (SAC 10M) ใช้เยอะ |
| **Storage** | 50 GB | 100 GB+ | Model checkpoints ทุก 200K steps + artifacts |
| **Template** | RunPod PyTorch | RunPod PyTorch 2.x | มี CUDA + cuDNN พร้อม |

---

## ขั้นตอนที่ 1 — สมัครบัญชี RunPod

1. ไปที่ [https://www.runpod.io](https://www.runpod.io)
2. กด **Sign Up** (ใช้ Google/GitHub/Email)
3. เข้า **Billing** → เติมเงิน (แนะนำ $10–20 สำหรับเริ่มต้น)
4. ไปที่ **Settings** → **SSH Public Keys** (จะใช้ในขั้นตอนถัดไป)

---

## ขั้นตอนที่ 2 — สร้าง SSH Key

SSH Key ใช้สำหรับเชื่อมต่อเข้า Pod อย่างปลอดภัย (ไม่ต้องใส่ password)

### Windows (PowerShell)

```powershell
# สร้าง SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# กด Enter 3 ครั้ง (ใช้ path default, ไม่ต้องตั้ง passphrase สำหรับเริ่มต้น)
# จะได้ไฟล์ 2 อัน:
#   C:\Users\<ชื่อ>\.ssh\id_ed25519       ← private key (ห้ามแชร์!)
#   C:\Users\<ชื่อ>\.ssh\id_ed25519.pub   ← public key (เอาไปใส่ RunPod)

# ดู public key
Get-Content $env:USERPROFILE\.ssh\id_ed25519.pub
```

### macOS / Linux

```bash
# สร้าง SSH key pair
ssh-keygen -t ed25519 -C "your_email@example.com"

# Enter 3 ครั้ง (default path, no passphrase)

# ดู public key
cat ~/.ssh/id_ed25519.pub
```

ผลที่ได้จะเป็นบรรทัดยาวๆ หน้าตาแบบนี้:
```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIxxxxxxxxxxxxxxxxxxxxxxxxxx your_email@example.com
```

> **สำคัญ**: ไฟล์ **private key** (`id_ed25519` ไม่มี .pub) ห้ามแชร์ให้ใครเด็ดขาด!

---

## ขั้นตอนที่ 3 — เพิ่ม Public Key ลง RunPod

1. ไปที่ RunPod → **Settings** → **SSH Public Keys**
2. กด **Add SSH Key**
3. ตั้งชื่อ (เช่น `my-laptop`)
4. วาง public key ที่ copy มาจากขั้นตอนที่ 2
5. กด **Save**

---

## ขั้นตอนที่ 4 — สร้าง Pod

1. ไปที่ **Pods** → กด **+ Deploy**
2. เลือก **GPU**: `RTX 5090` (หรือ RTX 4090/A100 ตามงบ)
3. เลือก **Template**: `RunPod Pytorch 2.x`
4. ตั้ง **Container Disk**: `50 GB`
5. ตั้ง **Volume Disk**: `50 GB` (persistent storage — ข้อมูลไม่หายเมื่อ stop pod)
6. กด **Deploy On-Demand** (หรือ Spot ถ้าต้องการราคาถูกกว่า)
7. รอ ~1-3 นาที จนสถานะเป็น **Running** ✅

### ⚠️ เรื่อง Spot vs On-Demand
- **On-Demand**: การันตีว่าไม่โดนเด้ง ราคาแพงกว่า
- **Spot**: ราคาถูกกว่า ~50% แต่อาจโดนเด้งถ้า demand สูง
- **แนะนำ**: ใช้ On-Demand ตอนเทรน (กลัวเทรนเสร็จครึ่งทางแล้วโดนเด้ง)

---

## ขั้นตอนที่ 5 — SSH เข้า Pod

1. บนหน้า Pods → กดที่ Pod ของคุณ → **Connect**
2. จะเห็นข้อมูล SSH:
   ```
   ssh root@<IP_ADDRESS> -p <PORT> -i ~/.ssh/id_ed25519
   ```
3. เปิด Terminal แล้วรัน:

### Windows (PowerShell)
```powershell
ssh root@123.456.789.0 -p 12345 -i $env:USERPROFILE\.ssh\id_ed25519
```

### macOS / Linux
```bash
ssh root@123.456.789.0 -p 12345 -i ~/.ssh/id_ed25519
```

4. ตอบ `yes` เมื่อถามเรื่อง fingerprint
5. ถ้าเข้าได้จะเห็น:
```
root@<pod-id>:/workspace#
```

---

## ขั้นตอนที่ 6 — VS Code Remote-SSH

แนะนำใช้ VS Code ต่อเข้า Pod เพื่อเขียนโค้ดและรัน notebook ได้สะดวก

### 6.1 ติดตั้ง Extension
- เปิด VS Code → Extensions → ค้นหา **Remote - SSH** → Install

### 6.2 ตั้งค่า SSH Config

เปิด/สร้างไฟล์ SSH config:

**Windows**: `C:\Users\<ชื่อ>\.ssh\config`  
**macOS/Linux**: `~/.ssh/config`

เพิ่มข้อมูล Pod:

```
Host runpod
    HostName 123.456.789.0
    Port 12345
    User root
    IdentityFile ~/.ssh/id_ed25519
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```

> **หมายเหตุ**: ทุกครั้งที่สร้าง Pod ใหม่ IP/Port จะเปลี่ยน ต้องอัปเดตไฟล์นี้

### 6.3 เชื่อมต่อ
1. กด `Ctrl+Shift+P` → `Remote-SSH: Connect to Host`
2. เลือก `runpod`
3. รอ VS Code ติดตั้ง server (ครั้งแรก ~1-2 นาที)
4. เปิด folder: `/workspace/robotic-sim-dishwash`

### 6.4 ติดตั้ง Extensions บน Remote
ต้องติดตั้ง extensions ซ้ำบน remote:
- **Python**
- **Jupyter**

---

## ขั้นตอนที่ 7 — Clone & Setup Project

เมื่อ SSH เข้า Pod แล้ว:

```bash
# 1. ย้ายไป workspace (persistent volume)
cd /workspace

# 2. Clone repo
git clone https://github.com/siriponsri/robotic-sim-dishwash.git
cd robotic-sim-dishwash

# 3. รัน setup script (สร้าง venv + ติดตั้ง dependencies + register kernel)
bash scripts/runpod_setup.sh /workspace/robotic-sim-dishwash
```

Script จะทำ 6 ขั้นตอนอัตโนมัติ:
```
[1/6] Creating virtual environment at /workspace/robotic-sim-dishwash/.env
[2/6] Activating environment
[3/6] Upgrading pip
[4/6] Installing project dependencies
[5/6] Registering Jupyter kernel
[6/6] Running smoke test
✅ RunPod environment is ready
```

### ตั้งค่า MLflow credentials
```bash
cp .env.example .env.local
nano .env.local
# ใส่ค่า MLFLOW_TRACKING_URI, USERNAME, PASSWORD
```

---

## ขั้นตอนที่ 8 — ตรวจสอบ GPU

```bash
source .env/bin/activate

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
    print('VRAM:', round(torch.cuda.get_device_properties(0).total_mem / 1e9, 1), 'GB')
"
```

ผลที่ต้องได้ (ตัวอย่าง):
```
PyTorch: 2.10.0+cu121
CUDA available: True
GPU: NVIDIA GeForce RTX 5090
VRAM: 32.0 GB
```

> ⚠️ ถ้า `CUDA available: False` → ลอง `nvidia-smi` ตรวจว่า GPU ทำงาน

---

## ขั้นตอนที่ 9 — รัน Notebook

### วิธีที่ 1: VS Code Remote (แนะนำ)
1. เปิด VS Code Remote-SSH (ขั้นตอนที่ 6)
2. เปิดไฟล์ `notebooks/NB06_train_ppo.ipynb`
3. เลือก kernel: `Python (.env robotic-sim-dishwash)`
4. รัน cell ทีละอัน

### วิธีที่ 2: JupyterLab
```bash
source .env/bin/activate
jupyter lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root
```
แล้วเปิด URL ที่แสดงในเบราว์เซอร์ (ต้อง port forward ถ้าใช้ SSH)

### ลำดับการรัน Notebook บน RunPod

```
1. NB01 — Smoke test (ตรวจว่า env ทั้ง 2 task ใช้ได้ + GPU ทำงาน)
2. NB02 — Env Exploration (สำรวจ Apple env)
3. NB03 — Reward & Safety (ทดสอบ reward + MLflow)
4. NB04 — Baselines (ได้ BaseController สำหรับ NB07)
5. NB05 — Train PPO      ← ใช้เวลา 2-4 ชม.
6. NB06 — Train SAC       ← ใช้เวลา 2-4 ชม.
7. NB07 — Residual SAC    ← ใช้เวลา 10-20 ชม. (5 beta values)
8. NB08 — Evaluation (เปรียบเทียบ 3 methods → ประกาศผู้ชนะ)
9. NB09 — Bonus DishWipe  ← ใช้เวลา 2-4 ชม. (winner only)
```

> **Tip**: NB05 และ NB06 ไม่ต้องพึ่งกัน — สามารถรันพร้อมกันได้ (ถ้ามี VRAM พอ)

---

## การ Sync Code กับ GitHub

### ก่อนเริ่มทำงานบน RunPod
```bash
cd /workspace/robotic-sim-dishwash
git pull origin main
```

### หลังจากเทรนเสร็จ — push ผลลัพธ์กลับ
```bash
git add artifacts/NB05/ artifacts/NB06/ artifacts/NB07/
git add artifacts/NB08/ artifacts/NB09/
git commit -m "feat: add trained models PPO/SAC/Residual + evaluation"
git push origin main
```

### Download model กลับเครื่อง local (ถ้าไม่อยากใช้ git)
```bash
# บนเครื่อง local
scp -P 12345 root@123.456.789.0:/workspace/robotic-sim-dishwash/artifacts/NB05/ppo_apple.zip ./
```

---

## Troubleshooting

### 1. `Permission denied (publickey)`

**สาเหตุ**: SSH key ไม่ตรงกัน  
**แก้**:
- ตรวจว่า public key บน RunPod ตรงกับ `~/.ssh/id_ed25519.pub`
- ตรวจว่า private key path ถูกต้อง
- ลองลบ key บน RunPod แล้วเพิ่มใหม่

### 2. `Connection refused` / `Connection timed out`

**สาเหตุ**: Pod ยังไม่ ready หรือ IP/Port ผิด  
**แก้**:
- รอ Pod เป็น Running ก่อน
- ตรวจ IP + Port จากหน้า Connect ของ RunPod (อาจเปลี่ยน)

### 3. `CUDA out of memory`

**สาเหตุ**: VRAM ไม่พอ  
**แก้**:
- ลด `N_ENVS` ใน config (เช่น จาก 64 เป็น 32)
- ลด `batch_size`
- ลด `buffer_size` (SAC)
- ปิด notebook อื่นที่ไม่ใช้ (ฆ่า kernel)

### 4. Pod ถูก Stop / Spot ถูกเด้ง

**แก้**:
- ข้อมูลใน `/workspace/` (Volume) จะยังอยู่ — Start pod ใหม่ได้
- ข้อมูลนอก `/workspace/` จะหาย
- แนะนำ: Save checkpoint บ่อยๆ + push ขึ้น git

### 5. `nvidia-smi` ไม่ทำงาน

**สาเหตุ**: Driver mismatch  
**แก้**: ลอง Pod template อื่น หรือ restart pod

### 6. Jupyter kernel ไม่เจอ

**แก้**:
```bash
source /workspace/robotic-sim-dishwash/.env/bin/activate
python -m ipykernel install --user --name robotic-sim-dishwash-env --display-name "Python (.env robotic-sim-dishwash)"
```

---

*ก่อนหน้า → [01 — Setup Local](01_repo_setup_local.md) | ต่อไป → [03 — Environment & Task](03_environment_and_task.md)*
