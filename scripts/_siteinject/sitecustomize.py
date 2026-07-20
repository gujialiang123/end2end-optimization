import os, sys
if os.environ.get("CUSTOM_MOE_INJECT", "0") == "1":
    try:
        sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
        import custom_moe_patch
        custom_moe_patch.install()
    except Exception as e:
        print(f"[sitecustomize] custom_moe_patch install failed: {e}", flush=True)
