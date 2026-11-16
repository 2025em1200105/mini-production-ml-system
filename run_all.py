"""
run_all.py
----------
One command to run my whole project from start to finish:

    python run_all.py

It runs every step in order and stops at the first failure.
I use 'sys.executable' so it uses the same Python / virtual environment on Windows and Mac.
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = [
    ("1. Download data from Kaggle + leak-free split", ["-m", "src.data_pipeline"]),
    ("2. Train baselines + models, tune threshold", ["-m", "src.train"]),
    ("3. Evaluate on the locked test set", ["-m", "src.evaluate"]),
    ("4. Batch scoring of new customers", ["-m", "serving.predict_batch",
                                           "--input", "data/processed/new_customers.csv",
                                           "--output", "reports/batch_predictions.csv"]),
    ("5. Drift check - normal data", ["-m", "src.drift", "--input", "data/processed/new_customers.csv"]),
    ("6. Drift check - simulated market shift", ["-m", "src.drift", "--input",
                                                 "data/processed/new_customers.csv", "--simulate-drift"]),
    ("7. Latency / throughput / edge-case benchmark", ["-m", "serving.benchmark"]),
    ("8. Unit tests", ["-m", "pytest", "-q"]),
]


def main():
    total = time.perf_counter()
    for title, args in STEPS:
        print("\n" + "=" * 70 + f"\n{title}\n" + "=" * 70, flush=True)
        t0 = time.perf_counter()
        result = subprocess.run([sys.executable, *args], cwd=ROOT)
        if result.returncode != 0:
            print(f"\nSTEP FAILED: {title}")
            sys.exit(result.returncode)
        print(f"-- finished in {time.perf_counter() - t0:.1f} s")
    print(f"\nALL STEPS PASSED in {time.perf_counter() - total:.1f} s. Results are in the 'reports' folder.")


if __name__ == "__main__":
    main()
