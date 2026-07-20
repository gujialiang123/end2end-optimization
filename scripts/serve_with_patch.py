"""Launch sglang server with custom_moe_patch installed before model load.
Usage: python scripts/serve_with_patch.py -- <sglang.launch_server args>
Env CUSTOM_MOE=1 enables the custom kernel path.
"""
import sys, runpy
sys.path.insert(0, "scripts")
import custom_moe_patch
custom_moe_patch.install()

# strip our own argv[0]; forward the rest to sglang.launch_server
if "--" in sys.argv:
    idx = sys.argv.index("--")
    fwd = sys.argv[idx + 1:]
else:
    fwd = sys.argv[1:]
sys.argv = ["sglang.launch_server"] + fwd
runpy.run_module("sglang.launch_server", run_name="__main__")
