import traceback

from rp_warmup_sd import warmup

try:
    warmup("flux-schnell")
except Exception as e:
    print(e)
    traceback.print_exc()
