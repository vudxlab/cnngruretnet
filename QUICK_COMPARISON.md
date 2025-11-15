# 🚀 SO SÁNH NHANH - CÁC OUTPUT_STEPS

Vì bạn đã có sẵn kết quả training trong folder `results/`, chỉ cần chạy script phân tích!

## Cấu trúc folder hiện tại

```
results/
├── 5/              # output_steps=5
│   ├── conv1d_gru/
│   │   └── metrics.csv
│   ├── gru/
│   │   └── metrics.csv
│   └── conv1d/
│       └── metrics.csv
├── 10/             # output_steps=10
│   ├── conv1d_gru/
│   ├── gru/
│   └── conv1d/
├── 15/
├── 20/
├── 30/
└── 40/
```

## ⚡ Chạy ngay (30 giây)

```bash
# Kích hoạt environment
conda activate tf

# Chạy script phân tích (chỉ metrics)
python analyze_existing_results.py --results_dir results --output_dir analysis

# Chạy với prediction comparisons (mất thêm 3-5 phút)
python analyze_existing_results.py --results_dir results --output_dir analysis --plot_predictions

# Hoặc nếu folder kết quả tên khác:
python analyze_existing_results.py --results_dir ten_folder_khac --output_dir analysis
```

## 📊 Kết quả sẽ tạo ra

Folder `analysis/` sẽ chứa:

### 1. `comparison_table.csv` - Bảng so sánh đầy đủ

```csv
output_step,model,dataset,rmse,mae,r2
5,conv1d_gru,Test,0.001234,0.000987,0.9756
5,gru,Test,0.001345,0.001045,0.9630
5,conv1d,Test,0.002301,0.001634,0.8674
10,conv1d_gru,Test,0.001456,0.001123,0.9634
...
```

### 2. `metrics_vs_output_steps.png` - Biểu đồ Line Charts

3 subplots:
- **R² vs Output Steps** (càng cao càng tốt)
- **RMSE vs Output Steps** (càng thấp càng tốt)
- **MAE vs Output Steps** (càng thấp càng tốt)

Mỗi line = 1 model (3 models)

**Insight:** Xem trend khi output_steps tăng:
- Performance có giảm không?
- Model nào stable nhất?
- Model nào degradation ít nhất?

### 3. `heatmaps.png` - Heatmaps

3 heatmaps (R², RMSE, MAE):
- **Rows:** Models (conv1d_gru, gru, conv1d)
- **Columns:** Output steps (5, 10, 15, 20, 30, 40)
- **Colors:**
  - Xanh = Tốt
  - Vàng = Trung bình
  - Đỏ = Kém

**Insight:** Một cái nhìn tổng quan nhanh về performance

### 4. `best_configurations.csv` - Best configs

```csv
Metric,Model,Output Steps,Value,RMSE,MAE
Best R²,conv1d_gru,5,0.9756,0.001234,0.000987
Best RMSE,conv1d_gru,5,0.001234,0.001234,0.000987
Best MAE,gru,10,0.000950,0.001345,0.000950
```

**Insight:** Cấu hình tốt nhất cho từng metric

### 5. `summary_report.txt` - Báo cáo text

### 6. `predictions_comparison/` - Prediction comparisons (nếu dùng --plot_predictions)

Folder chứa biểu đồ so sánh predictions:
- ⭐ `overlay_out{5,10,15,20,30,40}.png` - **Overlay 3 models (KHUYÊN XEM!)**
- `comparison_out{5,10,15,20,30,40}.png` - So sánh models (3 subplots)
- `comparison_{model}.png` - So sánh output_steps theo model
- `grid_sample{0,1,2}.png` - Grid tổng quan

**Format overlay:** Giống `prediction_sample_1.png` nhưng có CẢ 3 predictions overlay trên cùng subplot!

**Xem chi tiết:** `PREDICTION_COMPARISON_GUIDE.md`

```
==================================================================================================
BÁO CÁO SO SÁNH MODELS VỚI CÁC OUTPUT_STEPS
==================================================================================================

Tổng số models: 3
Models: CONV1D-GRU, CONV1D, GRU
Output steps: [5, 10, 15, 20, 30, 40]
Tổng số combinations: 18

==================================================================================================
PERFORMANCE SUMMARY BY MODEL (Test Set)
==================================================================================================

CONV1D-GRU:
    R² Range: 0.945600 → 0.975500
    RMSE Range: 0.001010 → 0.001567
    MAE Range: 0.000745 → 0.001123
    Best output_step: 5 (R²=0.975500, RMSE=0.001010)
    Worst output_step: 40 (R²=0.945600, RMSE=0.001567)
    Performance degradation: 3.07%

GRU:
    R² Range: 0.923400 → 0.963000
    ...

CONV1D:
    R² Range: 0.805600 → 0.867400
    ...

==================================================================================================
IMPACT OF OUTPUT_STEPS ON AVERAGE PERFORMANCE
==================================================================================================

Output Steps = 5:
    Average R²: 0.935367 ± 0.050234
    Average RMSE: 0.001615 ± 0.000575
    Average MAE: 0.001122 ± 0.000312
    Best model: CONV1D_GRU (R²=0.975500)

Output Steps = 10:
    Average R²: 0.918733 ± 0.048567
    ...

...
```

## 🎯 Câu hỏi thường gặp

### Q1: Model nào tốt nhất?

**A:** Xem `best_configurations.csv` hoặc biểu đồ line charts

### Q2: Output_steps nào tốt nhất?

**A:** Thường là output_steps nhỏ (5 hoặc 10) sẽ tốt nhất. Xem heatmaps để so sánh.

### Q3: Performance giảm bao nhiêu khi tăng output_steps?

**A:** Xem `summary_report.txt` phần "Performance degradation"

### Q4: Model nào stable nhất khi tăng output_steps?

**A:** Xem line charts - model nào có line "bằng phẳng" nhất

### Q5: Muốn xem predictions của best model?

**A:** Vào folder `results/5/conv1d_gru/predictions/` để xem prediction plots

## 💡 Tips

### So sánh 2 output_steps cụ thể

```bash
# Xem metrics của output_steps=5
cat results/5/conv1d_gru/metrics.csv

# Xem metrics của output_steps=20
cat results/20/conv1d_gru/metrics.csv
```

### Tìm model tốt nhất cho long-term forecasting

Xem heatmap - cột 30 hoặc 40, row nào xanh nhất?

### Kiểm tra một model cụ thể

```python
import pandas as pd

# Load comparison table
df = pd.read_csv('analysis/comparison_table.csv')

# Filter conv1d_gru
conv1d_gru_df = df[df['model'] == 'conv1d_gru']
print(conv1d_gru_df)
```

## 🏆 Kết quả ví dụ (dự kiến)

**Best Overall:** Conv1D-GRU với output_steps=5
- R² = 0.9755
- RMSE = 0.001010
- MAE = 0.000745

**Trend:**
- Output_steps tăng → Performance giảm ~3-10%
- Conv1D-GRU tốt nhất tất cả output_steps
- GRU là runner-up
- Conv1D kém hơn đáng kể

## 📸 Ví dụ Visualization

### Line Chart mẫu:
```
R² Score vs Output Steps
1.00 ┤         Conv1D-GRU ●──●──●──●──●──●
0.95 ┤                   GRU ■──■──■──■──■──■
0.90 ┤
0.85 ┤                         Conv1D ▲──▲──▲──▲──▲──▲
0.80 ┤
     └─────────────────────────────────────────
       5    10   15   20   30   40
```

### Heatmap mẫu:
```
         5      10     15     20     30     40
Conv1D-GRU  🟢 0.98  🟢 0.96  🟡 0.94  🟡 0.92  🟡 0.90  🟠 0.88
GRU         🟢 0.96  🟡 0.94  🟡 0.92  🟡 0.90  🟠 0.88  🟠 0.86
Conv1D      🟡 0.87  🟠 0.85  🟠 0.83  🔴 0.81  🔴 0.79  🔴 0.77
```

## ⚡ One-liner

```bash
# Chỉ metrics (30 giây)
conda activate tf && python analyze_existing_results.py

# Metrics + prediction comparisons (3-5 phút)
conda activate tf && python analyze_existing_results.py --plot_predictions
```

**Done!** 🎉

---

**Xem thêm:** Đọc `analysis/summary_report.txt` để có insights chi tiết!
