import traceback

from rp_warmup_sd import warmup

import os

try:
    warmup_model = os.getenv("WARMUP_MODEL", "flux-schnell")
    warmup(warmup_model)
except Exception as e:
    print(e)
    traceback.print_exc()
