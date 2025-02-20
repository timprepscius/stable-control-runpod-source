from rp_warmup_sd_ctrl import warmup
import traceback

try:
    warmup()
except Exception as e:
    print(e)
    traceback.print_exc()
