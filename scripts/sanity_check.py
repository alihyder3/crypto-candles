import importlib
import sys
import platform
from src.config import CONFIG

def try_import(name):
    try:
        mod = importlib.import_module(name)
        return mod.__version__ if hasattr(mod, "__version__") else "OK"
    except Exception as e:
        return f"ERROR: {e}"

def main():
    print("=== Sanity Check ===")
    print(f"Python: {sys.version.split()[0]}, Platform: {platform.platform()}")
    print(f"ML_FRAMEWORK={CONFIG.ml_framework}, WEB_FRAMEWORK={CONFIG.web_framework}")
    print(f"Defaults: SYMBOL={CONFIG.symbol}, INTERVAL={CONFIG.interval}, "
          f"LOOKBACK={CONFIG.lookback}, HORIZON={CONFIG.horizon}")

    print("\nPackages:")
    for pkg in ["numpy", "pandas", "python_binance", "requests", "websocket", "tensorflow", "torch", "Flask", "fastapi"]:
        ver = try_import(pkg)
        print(f" - {pkg}: {ver}")

    print("\nResult: If your chosen ML + web framework show proper versions (not ERROR) and numpy/pandas load, you're good.")

if __name__ == "__main__":
    main()
