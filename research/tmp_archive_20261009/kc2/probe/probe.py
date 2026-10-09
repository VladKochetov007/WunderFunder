import os
print("probe_ok", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else None, flush=True)
