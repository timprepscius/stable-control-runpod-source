from rp_warmup_sd_ctrl import warmup
import traceback

try:
    warmup("sdxl-lightning")
except Exception as e:
    print(e)
    traceback.print_exc()
