# 📁 PROJECT STRUCTURE - Clean & Organized

## 🎯 Cấu trúc Project sau khi dọn dẹp

### 📂 Core Python Modules

**Training & Data Processing:**
- `main.py` - Entry point chính, train models
- `config.py` - Configuration và hyperparameters
- `data_loader.py` - Load dữ liệu từ .mat files
- `data_preprocessing.py` - Preprocessing và augmentation
- `data_cache.py` - Cache system (tiết kiệm 30x thời gian)
- `trainer.py` - Training logic
- `evaluator.py` - Evaluation metrics
- `utils.py` - Utility functions
- `visualization.py` - Plotting functions

**Models:**
- `model.py` - Deep Learning models (Conv1D-GRU-ResNet, GRU, Conv1D)
- `baseline_models.py` - Baseline models (Linear, XGBoost, LightGBM)
- `train_all_models.py` - Script train tất cả models

**Analysis:**
- `analyze_existing_results.py` - Phân tích kết quả, tạo metrics & visualizations
- `plot_prediction_comparison.py` - Vẽ prediction comparison plots

### 📂 Documentation

**Quick Start:**
- `README.md` - Overview và quick start guide
- `quick_start.md` - Quick commands reference
- `USAGE.md` - Detailed usage instructions

**Guides:**
- `CACHE_GUIDE.md` - Hướng dẫn cache system
- `QUICK_COMPARISON.md` - Hướng dẫn so sánh models nhanh
- `PREDICTION_COMPARISON_GUIDE.md` - Hướng dẫn chi tiết prediction plots
- `OVERLAY_UPDATE_V2.md` - Cập nhật overlay comparison (10 best samples)

### 📂 Configuration Files

- `.gitignore` - Git ignore rules
- `requirements.txt` - Python dependencies
- `run.bat` - Quick run script (Windows)
- `run_analysis.bat` - Run analysis script (Windows)

### 📂 Data & Results

**Folders:**
```
4_Code/
├── Data/                    # Dữ liệu .mat
│   └── TH2_SETUP1.mat      # 50MB vibration data
├── cache/                   # Preprocessed data cache
│   └── data_*.pkl          # Cache files
├── results/                 # Training results
│   ├── 5/                  # output_steps=5
│   │   ├── conv1d_gru/     # Conv1D-GRU-ResNet results
│   │   ├── gru/            # GRU results
│   │   └── conv1d/         # Conv1D results
│   ├── 10/                 # output_steps=10
│   ├── 15/
│   └── 20/
└── analysis/                # Analysis outputs
    ├── comparison_table.csv
    ├── metrics_vs_output_steps.png
    ├── heatmaps.png
    ├── best_configurations.csv
    ├── summary_report.txt
    ├── training_curves/
    │   └── training_curves_out*.png
    └── predictions_comparison/
        ├── overlay_out*.png      # ⭐ 10 best samples
        ├── comparison_out*.png
        ├── comparison_*.png
        └── grid_sample*.png
```

## 🗑️ Files đã XÓA (Cleanup)

### Test Scripts (9 files):
- ❌ `test_overlay_best_samples.py`
- ❌ `test_training_curves.py`
- ❌ `benchmark_imports.py`
- ❌ `test_output/` (folder)

### Duplicate Scripts:
- ❌ `compare_output_steps.py` (chức năng trong main.py)
- ❌ `analyze_results.py` (duplicate analyze_existing_results.py)

### Outdated Documentation:
- ❌ `OVERLAY_COMPARISON_UPDATE.md` (có V2)
- ❌ `COMPARISON_GUIDE.md` (duplicate QUICK_COMPARISON)
- ❌ `PERFORMANCE_ANALYSIS.md` (informational only)
- ❌ `CLEANUP_SUMMARY.txt` (old)

**Total cleaned:** 9 files + 1 folder

## 📊 Project Stats

### Files Count:
- **Python modules:** 13 files
- **Documentation:** 7 files
- **Config files:** 4 files
- **Total:** 24 files (clean!)

### Lines of Code (estimated):
- Core modules: ~3,000 LOC
- Analysis scripts: ~1,500 LOC
- Documentation: ~2,000 lines

## 🚀 Workflows

### 1. Train Models
```bash
conda activate tf
python main.py --models conv1d_gru gru conv1d --output_steps 5
```

### 2. Analyze Results
```bash
conda activate tf
python -X utf8 analyze_existing_results.py --plot_predictions
```

### 3. Train All Models (Batch)
```bash
conda activate tf
python train_all_models.py
```

## 📝 Best Practices

### ✅ DO:
1. **Sử dụng cache** - Tiết kiệm 30x thời gian preprocessing
2. **Train multiple models** - `--models conv1d_gru gru conv1d`
3. **Analyze existing results** - Không cần re-train
4. **Xem overlay plots** - Dễ so sánh nhất

### ❌ DON'T:
1. **Không disable cache** trừ khi cần thiết
2. **Không xóa results/** - Mất kết quả training
3. **Không train lại** nếu đã có results

## 🔄 Update History

### v2.0 - Project Cleanup (Latest)
- ✅ Xóa 9 test/duplicate files
- ✅ Xóa outdated documentation
- ✅ Organized structure
- ✅ 10 best samples trong overlay
- ✅ Renamed Conv1D-GRU → Conv1D-GRU-ResNet

### v1.5 - Prediction Comparisons
- ✅ Overlay comparison plots
- ✅ Training curves comparison
- ✅ Grid overview

### v1.0 - Initial Release
- ✅ Core training pipeline
- ✅ 6 models support
- ✅ Cache system
- ✅ Analysis tools

## 📖 Documentation Guide

**Bắt đầu:**
1. Đọc `README.md` - Tổng quan
2. Đọc `quick_start.md` - Quick commands
3. Chạy `python main.py --help`

**Chi tiết:**
- Training: `USAGE.md`
- Cache: `CACHE_GUIDE.md`
- Analysis: `QUICK_COMPARISON.md`
- Predictions: `PREDICTION_COMPARISON_GUIDE.md`

**Updates:**
- Latest features: `OVERLAY_UPDATE_V2.md`

---

**Project clean & ready to use! 🎉**
