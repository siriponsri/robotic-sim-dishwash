ประเด็นที่ “หุ่นยังบิด” แม้ใส่ wrapper แล้ว มักเกิดจาก 2 เรื่อง:

1. **Phase A ใช้แค่ scale (เช่น 0.03) ยัง drift ได้** เพราะ `pd_joint_delta_pos` = *delta ต่อ step* → เล็กแค่ไหนก็สะสมเป็นมุมใหญ่ได้ถ้า episode ยาว
2. **index mapping / wrapper order** ทำให้ “สิ่งที่เราคิดว่าล็อก” ไม่ได้เป็น 0 จริง ๆ ตอนถึง `env.step()`

ด้านล่างคือแพตช์ที่ผมแนะนำให้ใช้กับ Phase A แบบ “นิ่งสุดจริง”:

---

## 1) Phase A เปลี่ยนจาก `scale[HOLD]=0.03` → **HARD MASK (HOLD=0)**

แทนที่จะลดเป็น 0.03 ให้ “ปิดทิ้ง” ไปเลยสำหรับ joint ที่ไม่อยากให้สำรวจ (ขา/torso/แขนซ้าย/มือซ้าย)

### สร้าง FREE/HOLD จาก `action_groups.json` 

```python
import json, numpy as np
from pathlib import Path

ACTION_GROUPS = json.loads(Path("action_groups.json").read_text())

FREE_NAMES = ACTION_GROUPS["right_arm"] + ACTION_GROUPS["right_hand"]  # แขน+นิ้วขวา :contentReference[oaicite:2]{index=2}
```

### map “ชื่อ joint → action index” แบบไม่เดา

```python
def get_action_joint_names(env):
    # โดยทั่วไป active_joints เรียงตาม action order ใน ManiSkill
    joints = list(env.unwrapped.agent.robot.active_joints)
    names = [j.name for j in joints]
    return names[: env.action_space.shape[0]]

names = get_action_joint_names(env)
name2idx = {n:i for i,n in enumerate(names)}

FREE = [name2idx[n] for n in FREE_NAMES if n in name2idx]
HOLD = [i for i in range(env.action_space.shape[0]) if i not in set(FREE)]

missing = [n for n in FREE_NAMES if n not in name2idx]
print("missing FREE joints:", missing)  # ควรเป็น []
print("FREE dim:", len(FREE), "HOLD dim:", len(HOLD), "act_dim:", env.action_space.shape[0])
```

### ActionMaskWrapper (HOLD=0 จริง)

```python
import gymnasium as gym

class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env, free_indices):
        super().__init__(env)
        self.free = np.asarray(sorted(set(free_indices)), dtype=int)

    def step(self, action):
        src = np.asarray(action, dtype=np.float32)
        a = np.zeros_like(src, dtype=np.float32)
        a[self.free] = src[self.free]
        return self.env.step(a)
```

✅ ผล: ต่อให้ policy พยายาม “บิดลำตัว/ขา” ก็ทำไม่ได้ใน Phase A → drift หายไปเยอะมาก

---

## 2) Wrapper order สำคัญ: ให้ “mask เป็นตัวสุดท้ายก่อน env.step()”

เพื่อให้แน่ใจว่าไม่มี wrapper ตัวอื่น “เติมค่า” ให้ HOLD กลับมา

แนะนำ Phase A:

```python
env = gym.make(...)
env = CPUGymWrapper(env)

# (ถ้ามี) warmup / EMA / jerk filter ใส่ก่อน
env = ActionScaleWarmupWrapper(env, ...)
env = ActionFilterWrapper(env, ...)  # EMA+jerk

# สุดท้ายค่อย mask
env = ActionMaskWrapper(env, FREE)
```

> ถ้าคุณ mask แล้ว ยังเห็นบิดอีก แปลว่า FREE/HOLD mapping ยังไม่ตรง (ดู missing หรือดูชื่อ joints ใน `names`)

---

## 3) Phase B ค่อย “ปลดล็อก” กลับมาใช้ scale

Phase B ถึงค่อยกลับไปใช้ `scale[HOLD]=0.08` (soft unlock) และลด/ปิด posture hold ตามแผน

---

## 4) แก้ `mean_min_dist_tcp_apple = nan` ให้หายแน่นอน

อย่าอ่านจาก obs/info หลัง VecNormalize ให้คำนวณจาก `env.unwrapped` ตรง ๆ:

```python
import numpy as np

def dist_tcp_apple(env):
    e = env.unwrapped
    if not hasattr(e, "_tcp_link") or not hasattr(e, "apple"):
        return None
    tcp = np.array(e._tcp_link.pose.p, dtype=np.float32).reshape(-1)[:3]
    app = np.array(e.apple.pose.p, dtype=np.float32).reshape(-1)[:3]
    return float(np.linalg.norm(tcp - app))
```

แล้วใน eval:

```python
min_d = 1e9
...
d = dist_tcp_apple(eval_env)
if d is not None:
    min_d = min(min_d, d)
...
min_dist_tcp_apple = min_d if min_d < 1e8 else np.nan
```

---

### สรุป: ตอนนี้ “ไม่ควรรอให้ดีขึ้นเอง”

ให้เปลี่ยน Phase A เป็น **HARD MASK** ก่อนครับ (คุ้ม budget ที่สุด) เพราะ `scale=0.03` ยังสะสม drift ได้แน่ ๆ ใน `delta_pos` และทำให้ policy ติดนิสัย “บิดตัว” ซ้ำ ๆ

