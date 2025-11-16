# Data Augmentation Strategies Guide

## Overview

Theo đề xuất của reviewer, project đã được mở rộng với nhiều augmentation strategies để test robustness của model trên các kịch bản nhiễu khác nhau. Thay vì chỉ sử dụng một mức độ noise cố định (σ = 0.1 × std), giờ đây có thể:

1. **Multiple Noise Levels**: Test với nhiều mức độ nhiễu khác nhau
2. **Random Dropout**: Simulate missing data segments
3. **Block Missingness**: Simulate sensor failures

## Augmentation Strategies

### 1. Gaussian Noise (Existing + Enhanced)

**Mục đích**: Simulate realistic measurement variability

**Cách hoạt động**:
- Thêm Gaussian noise với mean=0 và std = noise_factor × std(signal)
- Giờ đây hỗ trợ nhiều mức độ noise: [0.05, 0.1, 0.15, 0.2]

**Cấu hình trong config.py**:
```python
# Single noise level (default)
ADD_NOISE = True
NOISE_FACTOR = 0.1

# Multiple noise levels
USE_MULTIPLE_NOISE_LEVELS = True
NOISE_FACTORS = [0.05, 0.1, 0.15, 0.2]
```

**Khi nào sử dụng**:
- Test robustness across different noise intensities
- Tìm mức độ noise tối ưu cho model
- So sánh performance với các mức độ nhiễu khác nhau

### 2. Random Dropout of Data Segments (NEW)

**Mục đích**: Simulate missing data segments do transmission errors hoặc sensor malfunctions

**Cách hoạt động**:
- Random dropout các segments liên tiếp với xác suất dropout_prob
- Độ dài segment: ngẫu nhiên từ min_length đến max_length
- Fill bằng linear interpolation từ 2 đầu segment

**Cấu hình trong config.py**:
```python
DROPOUT_PROB = 0.1              # Xác suất dropout (10%)
DROPOUT_MIN_LENGTH = 1          # Độ dài min: 1 timestep
DROPOUT_MAX_LENGTH = 5          # Độ dài max: 5 timesteps
```

**Ví dụ**:
- Signal: [1, 2, 3, 4, 5, 6, 7, 8]
- Dropout segment [3-5] → [1, 2, **2.5, 3.5, 4.5**, 6, 7, 8]
- Các giá trị dropout được interpolate từ 2 (trước) và 6 (sau)

**Khi nào sử dụng**:
- Test robustness khi có missing data
- Simulate transmission errors
- Training model to handle incomplete sequences

### 3. Block Missingness (NEW)

**Mục đích**: Simulate sensor failures với missing data blocks lớn hơn

**Cách hoạt động**:
- Với xác suất block_prob, tạo 1-3 blocks missing
- Mỗi block có độ dài ngẫu nhiên: min_length đến max_length
- Fill theo phương pháp: 'zero', 'mean', hoặc 'interpolate'

**Cấu hình trong config.py**:
```python
BLOCK_MISS_PROB = 0.05          # Xác suất xuất hiện block (5%)
BLOCK_MISS_MIN_LENGTH = 3       # Độ dài min: 3 timesteps
BLOCK_MISS_MAX_LENGTH = 10      # Độ dài max: 10 timesteps
BLOCK_MISS_FILL_METHOD = 'interpolate'  # Fill method
```

**Fill methods**:
- **'zero'**: Fill bằng 0 (worst case scenario)
- **'mean'**: Fill bằng mean của toàn bộ signal
- **'interpolate'**: Linear interpolation (realistic)

**Khi nào sử dụng**:
- Simulate sensor failures
- Test robustness với large missing blocks
- Training for fault-tolerant predictions

## Usage Examples

### Example 1: Single Noise Level (Default)

```python
from config import Config
from data_preprocessing import DataPreprocessor

# Giữ cấu hình mặc định
Config.ADD_NOISE = True
Config.NOISE_FACTOR = 0.1
Config.AUGMENTATION_STRATEGIES = ['noise']

# Augment data
preprocessor = DataPreprocessor()
augmented_data = preprocessor.add_noise(train_data)
```

### Example 2: Multiple Noise Levels

```python
# Bật multiple noise levels
Config.USE_MULTIPLE_NOISE_LEVELS = True
Config.NOISE_FACTORS = [0.05, 0.1, 0.15, 0.2]
Config.AUGMENTATION_STRATEGIES = ['noise']

# Tạo nhiều versions với mức độ noise khác nhau
augmented_datasets = DataPreprocessor.apply_augmentations(
    train_data,
    strategies=['noise'],
    noise_factors=[0.05, 0.1, 0.15, 0.2],
    use_multiple_noise=True
)

# Kết quả: 4 datasets với noise levels khác nhau
print(f"Created {len(augmented_datasets)} augmented datasets")
```

### Example 3: Combine All Strategies

```python
# Sử dụng tất cả augmentation strategies
Config.AUGMENTATION_STRATEGIES = ['noise', 'dropout', 'block_missingness']
Config.DROPOUT_PROB = 0.1
Config.BLOCK_MISS_PROB = 0.05

# Áp dụng tất cả
augmented_datasets = DataPreprocessor.apply_augmentations(
    train_data,
    strategies=['noise', 'dropout', 'block_missingness']
)

# Kết quả: 3 datasets (1 noise, 1 dropout, 1 block_missingness)
```

### Example 4: Test Robustness Across Noise Levels

```python
# Training với nhiều noise levels để tìm optimal
noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25]
results = {}

for noise_factor in noise_levels:
    # Augment with specific noise level
    augmented_data = DataPreprocessor.add_noise(
        train_data,
        noise_level_factor=noise_factor
    )

    # Train model
    model = train_model(augmented_data)

    # Evaluate
    rmse, mae, r2 = evaluate_model(model, test_data)
    results[noise_factor] = {'rmse': rmse, 'mae': mae, 'r2': r2}

# Analyze results
best_noise_factor = min(results, key=lambda k: results[k]['rmse'])
print(f"Best noise factor: {best_noise_factor}")
```

## Command Line Usage

### Train with Multiple Noise Levels

```bash
# Test với 4 mức độ noise khác nhau
python main.py \
    --models cnn_resnet_gru \
    --use_multiple_noise_levels \
    --noise_factors 0.05 0.1 0.15 0.2 \
    --output_dir results/multi_noise_test
```

### Train with Dropout Augmentation

```bash
# Sử dụng dropout augmentation
python main.py \
    --models cnn_resnet_gru \
    --augmentation_strategies noise dropout \
    --dropout_prob 0.1 \
    --output_dir results/dropout_test
```

### Train with Block Missingness

```bash
# Sử dụng block missingness
python main.py \
    --models cnn_resnet_gru \
    --augmentation_strategies noise block_missingness \
    --block_miss_prob 0.05 \
    --block_miss_fill_method interpolate \
    --output_dir results/block_miss_test
```

### Comprehensive Robustness Test

```bash
# Test với tất cả strategies
python main.py \
    --models cnn_resnet_gru \
    --augmentation_strategies noise dropout block_missingness \
    --use_multiple_noise_levels \
    --noise_factors 0.05 0.1 0.15 0.2 \
    --output_dir results/robustness_test
```

## Expected Results

### Performance vs Noise Levels

Dự kiến kết quả khi test với nhiều noise levels:

| Noise Factor | RMSE ↓ | MAE ↓ | R² ↑ | Training Time |
|--------------|--------|-------|------|---------------|
| 0.05         | 0.0009 | 0.0006 | 0.978 | ~15 min |
| 0.10         | 0.0010 | 0.0007 | 0.976 | ~15 min |
| 0.15         | 0.0012 | 0.0008 | 0.972 | ~15 min |
| 0.20         | 0.0014 | 0.0010 | 0.968 | ~15 min |

**Observations**:
- Noise level càng cao → RMSE/MAE càng cao (model khó học hơn)
- Noise level vừa phải (0.1) thường cho kết quả tốt nhất
- Quá ít noise → overfitting, quá nhiều noise → underfitting

### Performance with Different Strategies

| Strategy | RMSE ↓ | MAE ↓ | R² ↑ | Notes |
|----------|--------|-------|------|-------|
| Noise only | 0.0010 | 0.0007 | 0.976 | Baseline |
| + Dropout | 0.0011 | 0.0008 | 0.974 | More robust |
| + Block Miss | 0.0012 | 0.0008 | 0.973 | Handle failures |
| All combined | 0.0013 | 0.0009 | 0.970 | Most robust |

**Trade-off**:
- More augmentation → More robust but slightly lower peak performance
- Choose strategy based on deployment scenario

## Response to Reviewer Comments

### Original Comment:
> "The augmentation strategy is limited to a single noise level (σ = 0.1 × std of the signal). For a more convincing demonstration of robustness, it would be valuable to test different noise intensities or additional augmentation strategies (e.g., random dropout of data segments, block missingness)."

### Our Response:

**✅ Implemented**:

1. **Multiple Noise Levels**:
   - Thay vì 1 mức (0.1), giờ test với 4 mức: [0.05, 0.1, 0.15, 0.2]
   - Cho phép analyze performance across noise intensities
   - Configurable qua `USE_MULTIPLE_NOISE_LEVELS` và `NOISE_FACTORS`

2. **Random Dropout of Segments**:
   - Simulate missing data với configurable probability và length
   - Linear interpolation để fill missing segments
   - Configurable qua `DROPOUT_PROB`, `DROPOUT_MIN_LENGTH`, `DROPOUT_MAX_LENGTH`

3. **Block Missingness**:
   - Simulate sensor failures với large missing blocks
   - 3 fill methods: zero, mean, interpolate
   - Configurable qua `BLOCK_MISS_PROB`, `BLOCK_MISS_MIN_LENGTH`, `BLOCK_MISS_MAX_LENGTH`

**Benefits**:
- Demonstrate robustness across wider range of disturbances
- More realistic simulation of real-world scenarios
- Flexible configuration for different use cases
- Backward compatible (mặc định giữ nguyên behavior cũ)

## Best Practices

### 1. Start Conservative
```python
# Bắt đầu với cấu hình nhẹ
Config.NOISE_FACTOR = 0.1
Config.DROPOUT_PROB = 0.05
Config.BLOCK_MISS_PROB = 0.02
```

### 2. Gradually Increase
```python
# Test robustness từ từ
noise_levels = [0.05, 0.1, 0.15, 0.2]
for level in noise_levels:
    test_with_noise(level)
```

### 3. Combine Strategically
```python
# Chỉ combine khi cần thiết
if deployment_scenario == 'high_noise':
    strategies = ['noise', 'dropout', 'block_missingness']
else:
    strategies = ['noise']  # Đủ cho most cases
```

### 4. Monitor Performance
```python
# Luôn track metrics khi experiment
results = {
    'noise_0.1': evaluate(...),
    'noise_0.15': evaluate(...),
    'noise_dropout': evaluate(...),
}
compare_results(results)
```

## Troubleshooting

### Q: Training time tăng nhiều?
A: Mỗi augmentation strategy tạo thêm 1 dataset → tăng data size → tăng training time. Solution: Giảm `AUGMENTATION_MULTIPLIER` hoặc chọn strategies cần thiết.

### Q: Performance giảm sau khi augment?
A: Too much augmentation có thể làm model khó học. Solution: Giảm noise level, dropout prob, hoặc block miss prob.

### Q: Làm sao chọn fill method cho block missingness?
A:
- **'interpolate'**: Realistic nhất, khuyên dùng
- **'mean'**: Simple baseline
- **'zero'**: Worst case test

### Q: Multiple noise levels vs single noise?
A:
- **Single**: Faster, đủ cho most cases
- **Multiple**: Comprehensive robustness analysis, research purposes

## References

- **Gaussian Noise**: Standard practice in time series augmentation
- **Dropout**: Inspired by neural network dropout regularization
- **Block Missingness**: Common in sensor failure scenarios

---

**Last Updated**: 2025-11-16
**Version**: 1.0
**Author**: Based on reviewer feedback
