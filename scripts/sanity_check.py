import importlib
import sys
import platform
from src.config import CONFIG

def try_import(pkg_label, import_name=None):
    modname = import_name or pkg_label
    try:
        mod = importlib.import_module(modname)
        ver = getattr(mod, "__version__", "OK")
        return ver
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("=== Sanity Check ===")
    print(f"Python: {sys.version.split()[0]}, Platform: {platform.platform()}")
    print(f"ML_FRAMEWORK={CONFIG.ml_framework}, WEB_FRAMEWORK={CONFIG.web_framework}")
    print(f"Defaults: SYMBOL={CONFIG.symbol}, INTERVAL={CONFIG.interval}, "
          f"LOOKBACK={CONFIG.lookback}, HORIZON={CONFIG.horizon}")

    print("\nPackages:")
    pkgs = [
        ("numpy", None),
        ("pandas", None),
        ("python-binance", "binance"),   # package name, import module
        ("requests", None),
        ("websocket-client", "websocket"),
        ("torch", None),
        ("fastapi", None),
        ("Flask", None),                 # okay if ERROR since we chose FastAPI
    ]
    for label, import_name in pkgs:
        ver = try_import(label, import_name)
        shown = f"{label} ({import_name or label})"
        print(f" - {shown}: {ver}")

    print("\nResult: Your chosen ML + web framework should show versions without ERROR.")

if __name__ == "__main__":
    main()
