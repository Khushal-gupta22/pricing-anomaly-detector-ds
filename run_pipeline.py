"""
Run the anomaly detection pipeline.
Usage:
    python run_pipeline.py                     # Full run
    python run_pipeline.py --no-prophet        # Skip Prophet (faster)
    python run_pipeline.py --sample 5000       # Quick test
"""
import sys
import os
import subprocess

# Ensure print output is visible immediately (no buffering)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(line_buffering=True)
os.environ['PYTHONUNBUFFERED'] = '1'

# Auto-activate venv if it exists and has a working Python
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(PROJECT_ROOT, ".venv", "Scripts", "python.exe")

if os.path.exists(VENV_PYTHON) and os.path.abspath(sys.executable) != os.path.abspath(VENV_PYTHON):
    # Test if venv python actually works before re-launching
    try:
        subprocess.run([VENV_PYTHON, "-c", "import sys"], timeout=10,
                       check=True, capture_output=True)
        print("[run_pipeline] Re-launching with venv Python...", flush=True)
        result = subprocess.run(
            [VENV_PYTHON, "-u", __file__] + sys.argv[1:],
            cwd=PROJECT_ROOT,
        )
        sys.exit(result.returncode)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError):
        pass  # Venv broken — continue with system Python

print(f"[run_pipeline] Starting (Python: {sys.executable})", flush=True)
sys.path.insert(0, PROJECT_ROOT)

try:
    print("[run_pipeline] Importing pipeline...", flush=True)
    from src.pipeline import main
    print("[run_pipeline] Import successful, launching pipeline...", flush=True)
    main()
except ImportError as e:
    print(f"\n[ERROR] Missing dependency: {e}", flush=True)
    print(f"Fix: pip install -r requirements.txt --user", flush=True)
    sys.exit(1)
except Exception as e:
    print(f"\n[ERROR] Pipeline failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)
