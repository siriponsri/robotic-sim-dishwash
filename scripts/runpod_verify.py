import importlib
import sys

packages = [
    "gymnasium",
    "mani_skill",
    "torch",
    "stable_baselines3",
    "numpy",
    "matplotlib",
    "mediapy",
    "IPython",
]

failed = []
for package in packages:
    try:
        importlib.import_module(package)
    except Exception as exc:
        failed.append((package, str(exc)))

if failed:
    print("[FAIL] Import check failed:")
    for name, message in failed:
        print(f"  - {name}: {message}")
    sys.exit(1)

import torch  # noqa: E402

print("[OK] All core imports succeeded")
print(f"[OK] Torch version: {torch.__version__}")
print(f"[OK] CUDA available: {torch.cuda.is_available()}")
