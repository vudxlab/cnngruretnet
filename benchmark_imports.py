"""
Script kiểm tra thời gian import và khởi động
Tìm nguyên nhân delay khi chạy main.py
"""

import time
import sys

def measure_import(module_name, import_statement):
    """Đo thời gian import một module"""
    start = time.time()
    exec(import_statement)
    elapsed = time.time() - start
    print(f"{module_name:<30} {elapsed:>8.3f}s")
    return elapsed

print("=" * 60)
print("BENCHMARK - IMPORT TIMES")
print("=" * 60)
print(f"{'Module':<30} {'Time'}")
print("-" * 60)

total = 0

# Python standard library
total += measure_import("argparse", "import argparse")
total += measure_import("sys", "import sys")
total += measure_import("os", "import os")
total += measure_import("datetime", "from datetime import datetime")

# NumPy
total += measure_import("numpy", "import numpy as np")

# SciPy
total += measure_import("scipy", "import scipy")
total += measure_import("scipy.io", "import scipy.io")

# Matplotlib
total += measure_import("matplotlib", "import matplotlib")
total += measure_import("matplotlib.pyplot", "import matplotlib.pyplot as plt")

# Pandas
total += measure_import("pandas", "import pandas as pd")

# Scikit-learn
total += measure_import("sklearn", "import sklearn")
total += measure_import("sklearn.metrics", "from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score")

# TensorFlow (usually the slowest)
total += measure_import("tensorflow", "import tensorflow as tf")

# XGBoost and LightGBM
try:
    total += measure_import("xgboost", "import xgboost")
except:
    print(f"{'xgboost':<30} {'NOT INSTALLED'}")

try:
    total += measure_import("lightgbm", "import lightgbm")
except:
    print(f"{'lightgbm':<30} {'NOT INSTALLED'}")

print("-" * 60)
print(f"{'TOTAL':<30} {total:>8.3f}s")
print("=" * 60)

# Now test importing project modules
print("\n" + "=" * 60)
print("PROJECT MODULES IMPORT TIMES")
print("=" * 60)
print(f"{'Module':<30} {'Time'}")
print("-" * 60)

project_total = 0

project_total += measure_import("config", "from config import Config")
project_total += measure_import("data_loader", "from data_loader import load_vibration_data")
project_total += measure_import("data_preprocessing", "from data_preprocessing import preprocess_data")
project_total += measure_import("model", "from model import create_model")
project_total += measure_import("trainer", "from trainer import train_model")
project_total += measure_import("evaluator", "from evaluator import evaluate_model")
project_total += measure_import("visualization", "from visualization import Visualizer")
project_total += measure_import("utils", "from utils import set_random_seed, create_output_directory")

print("-" * 60)
print(f"{'PROJECT TOTAL':<30} {project_total:>8.3f}s")
print("=" * 60)

print(f"\n🔍 ANALYSIS:")
print(f"  Library imports: {total:.3f}s")
print(f"  Project imports: {project_total:.3f}s")
print(f"  GRAND TOTAL:     {total + project_total:.3f}s")

if total > 10:
    print(f"\n⚠️  TensorFlow import is likely the bottleneck ({total:.1f}s)")
    print("   This is normal for TensorFlow's first import.")
    print("   Subsequent runs will be faster due to caching.")
