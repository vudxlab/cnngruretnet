"""
Script vẽ biểu đồ so sánh predictions giữa các models và output_steps
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Vẽ biểu đồ so sánh predictions"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results',
        help='Thư mục chứa kết quả (default: results/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis',
        help='Thư mục lưu kết quả (default: analysis/)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Số samples để vẽ (default: 5)'
    )
    return parser.parse_args()


def load_test_data_and_predictions(results_dir, output_step, model):
    """
    Load test data và predictions từ folder kết quả

    Args:
        results_dir: Thư mục chứa kết quả
        output_step: Output step
        model: Tên model

    Returns:
        tuple: (X_test, y_test, y_pred, scaler_values) hoặc None nếu không tìm thấy
    """
    model_path = os.path.join(results_dir, str(output_step), model)

    # Kiểm tra folder tồn tại
    if not os.path.exists(model_path):
        return None

    try:
        # Load scaler
        scaler_path = os.path.join(model_path, 'scaler_values.npy')
        if os.path.exists(scaler_path):
            scaler_values = np.load(scaler_path, allow_pickle=True).item()
        else:
            scaler_values = None

        # Cần load model và predict lại vì không lưu predictions
        # Hoặc có thể load từ predictions folder nếu có
        predictions_folder = os.path.join(model_path, 'predictions')

        if not os.path.exists(predictions_folder):
            print(f"  ⚠️  Không tìm thấy predictions folder: {model}/{output_step}")
            return None

        # Tìm các file numpy trong predictions folder (nếu có)
        # Thông thường predictions được lưu trong folder này
        # Nhưng cần re-generate từ model nếu chưa có

        return None  # Placeholder, cần implement load predictions

    except Exception as e:
        print(f"  ❌ Lỗi khi load data: {e}")
        return None


def regenerate_predictions(results_dir, output_step, model):
    """
    Re-generate predictions từ saved model

    Args:
        results_dir: Thư mục chứa kết quả
        output_step: Output step
        model: Tên model

    Returns:
        tuple: (y_true, y_pred) hoặc None
    """
    import tensorflow as tf
    from data_cache import DataCache
    from config import Config

    model_path = os.path.join(results_dir, str(output_step), model)

    try:
        # Load model
        model_file = os.path.join(model_path, 'model_saved.keras')
        if not os.path.exists(model_file):
            return None

        keras_model = tf.keras.models.load_model(model_file)

        # Load data from cache
        cache = DataCache()
        cache_key = cache.get_cache_key(
            sensor_idx=0,  # Assume sensor 0
            output_steps=output_step,
            add_noise=True,  # Assume with noise
            input_steps=50
        )

        if not cache.cache_exists(cache_key):
            print(f"  ⚠️  Không tìm thấy cache cho output_step={output_step}")
            return None

        data_dict = cache.load_cache(cache_key)

        # Get test data
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        preprocessor = data_dict['preprocessor']

        # Predict
        y_pred_scaled = keras_model.predict(X_test[:10], verbose=0)

        # Denormalize
        y_true = preprocessor.inverse_transform(y_test[:10])
        y_pred = preprocessor.inverse_transform(y_pred_scaled)

        return y_true, y_pred

    except Exception as e:
        print(f"  ❌ Lỗi regenerate predictions: {e}")
        return None


def regenerate_predictions_full(results_dir, output_step, model):
    """
    Re-generate predictions từ saved model cho TOÀN BỘ test set

    Args:
        results_dir: Thư mục chứa kết quả
        output_step: Output step
        model: Tên model

    Returns:
        tuple: (y_true, y_pred) toàn bộ test set hoặc None
    """
    import tensorflow as tf
    from data_cache import DataCache
    from config import Config
    from data_loader import VibrationDataLoader
    from data_preprocessing import DataPreprocessor

    model_path = os.path.join(results_dir, str(output_step), model)

    try:
        # Load model
        model_file = os.path.join(model_path, 'model_saved.keras')
        if not os.path.exists(model_file):
            return None

        keras_model = tf.keras.models.load_model(model_file)

        # Load data from cache hoặc tạo mới
        cache = DataCache()
        cache_key = cache.get_cache_key(
            sensor_idx=0,
            output_steps=output_step,
            add_noise=True,
            input_steps=50
        )

        if cache.cache_exists(cache_key):
            data_dict = cache.load_cache(cache_key)
        else:
            # Tạo cache mới
            mat_file = Config.get_mat_file_path()
            data_loader = VibrationDataLoader(mat_file)
            full_data = data_loader.load_mat_file()
            raw_data = data_loader.get_sensor_data(sensor_idx=0)

            preprocessor = DataPreprocessor(
                input_steps=50,
                output_steps=output_step,
                add_noise=True
            )

            data_dict = preprocessor.prepare_data(raw_data)
            cache.save_cache(data_dict, cache_key)

        # Get TOÀN BỘ test data
        X_test = data_dict['X_test']
        y_test = data_dict['y_test']
        preprocessor = data_dict['preprocessor']

        # Predict toàn bộ
        y_pred_scaled = keras_model.predict(X_test, verbose=0)

        # Denormalize
        y_true = preprocessor.inverse_transform(y_test)
        y_pred = preprocessor.inverse_transform(y_pred_scaled)

        return y_true, y_pred

    except Exception as e:
        print(f"  ❌ Lỗi regenerate full predictions: {e}")
        return None


def plot_comparison_by_output_step(results_dir, output_step, models, output_dir, num_samples=3):
    """
    Vẽ biểu đồ so sánh predictions của các models cho cùng output_step

    Args:
        results_dir: Thư mục kết quả
        output_step: Output step cần so sánh
        models: Danh sách models
        output_dir: Thư mục output
        num_samples: Số samples để vẽ
    """
    print(f"\n📊 Đang vẽ comparison cho output_step={output_step}...")

    # Load predictions cho tất cả models
    predictions_data = {}

    for model in models:
        result = regenerate_predictions(results_dir, output_step, model)
        if result is not None:
            y_true, y_pred = result
            predictions_data[model] = {
                'y_true': y_true,
                'y_pred': y_pred
            }
            print(f"  ✓ Loaded {model}")

    if not predictions_data:
        print(f"  ⚠️  Không có predictions cho output_step={output_step}")
        return

    # Vẽ biểu đồ
    num_models = len(predictions_data)
    fig, axes = plt.subplots(num_samples, num_models, figsize=(5*num_models, 3*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_models == 1:
        axes = axes.reshape(-1, 1)

    colors = {'conv1d_gru': '#2ecc71', 'gru': '#3498db', 'conv1d': '#e74c3c'}
    model_names = {'conv1d_gru': 'Conv1D-GRU-ResNet', 'gru': 'GRU', 'conv1d': 'Conv1D'}

    # Get y_true (same for all models)
    y_true_ref = list(predictions_data.values())[0]['y_true']

    for sample_idx in range(num_samples):
        for model_idx, (model, data) in enumerate(predictions_data.items()):
            ax = axes[sample_idx, model_idx]

            y_true_sample = data['y_true'][sample_idx]
            y_pred_sample = data['y_pred'][sample_idx]

            # Plot
            time_steps = range(len(y_true_sample))

            ax.plot(time_steps, y_true_sample, 'o-', linewidth=2, markersize=6,
                   label='True', color='black', alpha=0.7)
            ax.plot(time_steps, y_pred_sample, 's-', linewidth=2, markersize=6,
                   label='Predicted', color=colors.get(model, '#95a5a6'))

            # Calculate error
            mae = np.mean(np.abs(y_true_sample - y_pred_sample))

            model_display_name = model_names.get(model, model.upper().replace("_", "-"))
            ax.set_title(f'{model_display_name}\nSample {sample_idx+1} (MAE={mae:.6f})',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Timestep', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')

    plt.suptitle(f'Prediction Comparison - Output Steps = {output_step}',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.join(output_dir, 'predictions_comparison'), exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions_comparison', f'comparison_out{output_step}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Đã lưu: {output_file}")


def plot_comparison_by_model(results_dir, model, output_steps, output_dir, num_samples=3):
    """
    Vẽ biểu đồ so sánh predictions của cùng model với các output_steps khác nhau

    Args:
        results_dir: Thư mục kết quả
        model: Model cần so sánh
        output_steps: Danh sách output_steps
        output_dir: Thư mục output
        num_samples: Số samples để vẽ
    """
    model_names = {'conv1d_gru': 'Conv1D-GRU-ResNet', 'gru': 'GRU', 'conv1d': 'Conv1D'}
    print(f"\n📊 Đang vẽ comparison cho model={model}...")

    # Load predictions cho tất cả output_steps
    predictions_data = {}

    for out_step in output_steps:
        result = regenerate_predictions(results_dir, out_step, model)
        if result is not None:
            y_true, y_pred = result
            predictions_data[out_step] = {
                'y_true': y_true,
                'y_pred': y_pred
            }
            print(f"  ✓ Loaded output_step={out_step}")

    if not predictions_data:
        print(f"  ⚠️  Không có predictions cho model={model}")
        return

    # Vẽ biểu đồ
    num_output_steps = len(predictions_data)
    fig, axes = plt.subplots(num_samples, num_output_steps, figsize=(4*num_output_steps, 3*num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, -1)
    if num_output_steps == 1:
        axes = axes.reshape(-1, 1)

    for sample_idx in range(num_samples):
        for out_idx, (out_step, data) in enumerate(predictions_data.items()):
            ax = axes[sample_idx, out_idx]

            # Get data (chỉ lấy số timesteps = output_step)
            y_true_sample = data['y_true'][sample_idx][:out_step]
            y_pred_sample = data['y_pred'][sample_idx][:out_step]

            # Plot
            time_steps = range(len(y_true_sample))

            ax.plot(time_steps, y_true_sample, 'o-', linewidth=2, markersize=6,
                   label='True', color='black', alpha=0.7)
            ax.plot(time_steps, y_pred_sample, 's-', linewidth=2, markersize=6,
                   label='Predicted', color='#e74c3c')

            # Calculate error
            mae = np.mean(np.abs(y_true_sample - y_pred_sample))

            ax.set_title(f'Out={out_step} steps\nSample {sample_idx+1} (MAE={mae:.6f})',
                        fontsize=11, fontweight='bold')
            ax.set_xlabel('Timestep', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(fontsize=9, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')

    model_display_name = model_names.get(model, model.upper().replace("_", "-"))
    plt.suptitle(f'Prediction Comparison - {model_display_name}',
                fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.join(output_dir, 'predictions_comparison'), exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions_comparison', f'comparison_{model}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Đã lưu: {output_file}")


def plot_overlay_comparison(results_dir, output_step, models, output_dir, num_samples=10):
    """
    Vẽ biểu đồ overlay: Cả 3 models trên cùng một subplot
    Chọn 10 samples có MSE thấp nhất từ model Conv1D-GRU-ResNet

    Args:
        results_dir: Thư mục kết quả
        output_step: Output step cần so sánh
        models: Danh sách models
        output_dir: Thư mục output
        num_samples: Số samples để vẽ (default: 10)
    """
    print(f"\n📊 Đang vẽ overlay comparison cho output_step={output_step}...")

    # Load predictions cho tất cả models (toàn bộ test set để tìm best samples)
    predictions_data_full = {}

    for model in models:
        result = regenerate_predictions_full(results_dir, output_step, model)
        if result is not None:
            y_true, y_pred = result
            predictions_data_full[model] = {
                'y_true': y_true,
                'y_pred': y_pred
            }
            print(f"  ✓ Loaded {model}: {len(y_true)} samples")

    if not predictions_data_full:
        print(f"  ⚠️  Không có predictions cho output_step={output_step}")
        return

    # Tìm 10 samples có MSE thấp nhất từ Conv1D-GRU-ResNet
    conv1d_gru_key = None
    for key in predictions_data_full.keys():
        if 'conv1d_gru' in key.lower():
            conv1d_gru_key = key
            break

    if conv1d_gru_key is None:
        print(f"  ⚠️  Không tìm thấy Conv1D-GRU-ResNet model")
        # Fallback: use first num_samples
        best_indices = list(range(min(num_samples, len(list(predictions_data_full.values())[0]['y_true']))))
    else:
        # Tính MSE cho từng sample
        y_true = predictions_data_full[conv1d_gru_key]['y_true']
        y_pred = predictions_data_full[conv1d_gru_key]['y_pred']

        mse_per_sample = []
        for i in range(len(y_true)):
            mse = np.mean((y_true[i] - y_pred[i]) ** 2)
            mse_per_sample.append((i, mse))

        # Sort by MSE và lấy top num_samples
        mse_per_sample.sort(key=lambda x: x[1])
        best_indices = [idx for idx, _ in mse_per_sample[:num_samples]]

        print(f"  ✓ Đã chọn {num_samples} samples tốt nhất (MSE thấp nhất)")
        print(f"    Best MSE range: {mse_per_sample[0][1]:.6f} - {mse_per_sample[num_samples-1][1]:.6f}")

    # Extract predictions cho best samples
    predictions_data = {}
    for model, data in predictions_data_full.items():
        predictions_data[model] = {
            'y_true': data['y_true'][best_indices],
            'y_pred': data['y_pred'][best_indices]
        }

    # Load past data (input) từ cache cho best indices
    try:
        from data_cache import DataCache
        from config import Config
        from data_loader import VibrationDataLoader
        from data_preprocessing import DataPreprocessor

        cache = DataCache()
        cache_key = cache.get_cache_key(
            sensor_idx=0,
            output_steps=output_step,
            add_noise=True,
            input_steps=50
        )

        if cache.cache_exists(cache_key):
            data_dict = cache.load_cache(cache_key)
        else:
            # Tạo cache nếu chưa có
            mat_file = Config.get_mat_file_path()
            data_loader = VibrationDataLoader(mat_file)
            full_data = data_loader.load_mat_file()
            raw_data = data_loader.get_sensor_data(sensor_idx=0)
            preprocessor = DataPreprocessor(input_steps=50, output_steps=output_step, add_noise=True)
            data_dict = preprocessor.prepare_data(raw_data)
            cache.save_cache(data_dict, cache_key)

        X_test = data_dict['X_test']
        preprocessor = data_dict['preprocessor']

        # Denormalize past data cho best indices
        past_data_list = []
        for idx in best_indices:
            past_denorm = preprocessor.inverse_transform(X_test[idx].reshape(1, -1))
            past_data_list.append(past_denorm.flatten())
    except Exception as e:
        print(f"  ⚠️  Không load được past data: {e}")
        past_data_list = None

    # Colors cho từng model
    colors = {
        'conv1d_gru': '#2ecc71',  # Xanh lá - Conv1D-GRU-ResNet
        'gru': '#3498db',          # Xanh dương - GRU
        'conv1d': '#e74c3c'        # Đỏ - Conv1D
    }

    # Model name mapping
    model_names = {
        'conv1d_gru': 'Conv1D-GRU-ResNet',
        'gru': 'GRU',
        'conv1d': 'Conv1D'
    }

    # Tạo subplots (num_samples rows, 1 column)
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4*num_samples))

    if num_samples == 1:
        axes = [axes]

    # Get y_true reference (same for all models)
    y_true_ref = list(predictions_data.values())[0]['y_true']

    for sample_idx in range(num_samples):
        ax = axes[sample_idx]

        # Plot past data nếu có
        if past_data_list is not None and sample_idx < len(past_data_list):
            past_data = past_data_list[sample_idx]
            past_timesteps = range(len(past_data))
            ax.plot(past_timesteps, past_data, 'o-', linewidth=2, markersize=4,
                   label='Past Data (Input)', color='green', alpha=0.6)

        # Plot actual future
        y_true_sample = y_true_ref[sample_idx]
        future_start = len(past_data_list[sample_idx]) if past_data_list else 0
        future_timesteps = range(future_start, future_start + len(y_true_sample))

        ax.plot(future_timesteps, y_true_sample, 'o-', linewidth=2.5, markersize=6,
               label='Actual Future', color='blue', alpha=0.8, zorder=10)

        # Plot predictions từ tất cả models
        for model, data in predictions_data.items():
            y_pred_sample = data['y_pred'][sample_idx]

            model_display_name = model_names.get(model, model.upper().replace("_", "-"))
            ax.plot(future_timesteps, y_pred_sample, 's--', linewidth=2, markersize=5,
                   label=f'Predicted ({model_display_name})',
                   color=colors.get(model, '#95a5a6'), alpha=0.7)

        # Formatting
        ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample {sample_idx+1} - Output Steps = {output_step}',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle(f'Prediction Comparison (Overlay) - Output Steps = {output_step}',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.join(output_dir, 'predictions_comparison'), exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions_comparison', f'overlay_out{output_step}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Đã lưu: {output_file}")


def plot_all_combinations_grid(results_dir, models, output_steps, output_dir, sample_idx=0):
    """
    Vẽ grid tổng quan tất cả combinations (models × output_steps)

    Args:
        results_dir: Thư mục kết quả
        models: Danh sách models
        output_steps: Danh sách output_steps
        output_dir: Thư mục output
        sample_idx: Index của sample cần vẽ
    """
    print(f"\n📊 Đang vẽ grid tổng quan (sample {sample_idx})...")

    num_models = len(models)
    num_steps = len(output_steps)

    fig, axes = plt.subplots(num_models, num_steps, figsize=(3.5*num_steps, 3*num_models))

    if num_models == 1:
        axes = axes.reshape(1, -1)
    if num_steps == 1:
        axes = axes.reshape(-1, 1)

    colors = {'conv1d_gru': '#2ecc71', 'gru': '#3498db', 'conv1d': '#e74c3c'}
    model_names = {'conv1d_gru': 'Conv1D-GRU-ResNet', 'gru': 'GRU', 'conv1d': 'Conv1D'}

    for model_idx, model in enumerate(models):
        for step_idx, out_step in enumerate(output_steps):
            ax = axes[model_idx, step_idx]

            # Load predictions
            result = regenerate_predictions(results_dir, out_step, model)

            if result is None:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.set_title(f'{model}\nout={out_step}', fontsize=10)
                continue

            y_true, y_pred = result
            y_true_sample = y_true[sample_idx][:out_step]
            y_pred_sample = y_pred[sample_idx][:out_step]

            # Plot
            time_steps = range(len(y_true_sample))

            ax.plot(time_steps, y_true_sample, 'o-', linewidth=1.5, markersize=4,
                   label='True', color='black', alpha=0.7)
            ax.plot(time_steps, y_pred_sample, 's-', linewidth=1.5, markersize=4,
                   label='Pred', color=colors.get(model, '#95a5a6'))

            # Calculate MAE
            mae = np.mean(np.abs(y_true_sample - y_pred_sample))

            # Title
            model_display_name = model_names.get(model, model.upper().replace("_", "-"))
            if step_idx == 0:
                title = f'{model_display_name}\nout={out_step}\nMAE={mae:.4f}'
            else:
                title = f'out={out_step}\nMAE={mae:.4f}'

            ax.set_title(title, fontsize=9, fontweight='bold')

            if model_idx == num_models - 1:
                ax.set_xlabel('Step', fontsize=8)
            if step_idx == 0:
                ax.set_ylabel('Value', fontsize=8)

            if model_idx == 0 and step_idx == 0:
                ax.legend(fontsize=7, loc='best')

            ax.grid(True, alpha=0.2, linestyle='--')
            ax.tick_params(labelsize=7)

    plt.suptitle(f'Predictions Grid - All Combinations (Sample {sample_idx+1})',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    # Save
    os.makedirs(os.path.join(output_dir, 'predictions_comparison'), exist_ok=True)
    output_file = os.path.join(output_dir, 'predictions_comparison', f'grid_sample{sample_idx}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Đã lưu: {output_file}")


def main():
    """Main function"""
    args = parse_args()

    print("=" * 100)
    print("  VẼ BIỂU ĐỒ SO SÁNH PREDICTIONS")
    print("=" * 100)

    # Kiểm tra results_dir
    if not os.path.exists(args.results_dir):
        print(f"\n❌ Không tìm thấy thư mục: {args.results_dir}")
        sys.exit(1)

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Detect models và output_steps
    models = []
    output_steps = []

    for folder in os.listdir(args.results_dir):
        folder_path = os.path.join(args.results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        # Check if folder name is number (output_step)
        try:
            out_step = int(folder)
            output_steps.append(out_step)

            # Get models in this folder
            for model_folder in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, model_folder)):
                    if model_folder not in models:
                        models.append(model_folder)
        except ValueError:
            continue

    models = sorted(models)
    output_steps = sorted(output_steps)

    print(f"\nĐã phát hiện:")
    print(f"  Models: {models}")
    print(f"  Output steps: {output_steps}")

    if not models or not output_steps:
        print("\n❌ Không tìm thấy dữ liệu!")
        sys.exit(1)

    # Vẽ các biểu đồ
    print("\n" + "=" * 100)
    print("  ĐANG TẠO VISUALIZATIONS")
    print("=" * 100)

    # 1. Overlay comparison (CẢ 3 models trên cùng subplot - KHUYÊN DÙNG)
    print("\n1. Overlay Comparison (3 models on same plot):")
    for out_step in output_steps:
        plot_overlay_comparison(args.results_dir, out_step, models,
                               args.output_dir, num_samples=args.num_samples)

    # 2. Comparison by output_step (so sánh models cho mỗi output_step - 3 subplots)
    print("\n2. Comparison by Output Step (separate subplots):")
    for out_step in output_steps:
        plot_comparison_by_output_step(args.results_dir, out_step, models,
                                      args.output_dir, num_samples=args.num_samples)

    # 3. Comparison by model (so sánh output_steps cho mỗi model)
    print("\n3. Comparison by Model:")
    for model in models:
        plot_comparison_by_model(args.results_dir, model, output_steps,
                                args.output_dir, num_samples=args.num_samples)

    # 4. Grid tổng quan
    print("\n4. Overview Grid:")
    for sample_idx in range(min(3, args.num_samples)):
        plot_all_combinations_grid(args.results_dir, models, output_steps,
                                   args.output_dir, sample_idx=sample_idx)

    print("\n" + "=" * 100)
    print("✅ HOÀN THÀNH!")
    print("=" * 100)
    print(f"\n📁 Kết quả lưu tại: {args.output_dir}/predictions_comparison/")
    print("\nCác files đã tạo:")
    print("  🌟 overlay_out{5,10,15,20,30,40}.png     # Overlay 3 models (KHUYÊN XEM)")
    print("  - comparison_out{5,10,15,20,30,40}.png  # So sánh models (3 subplots)")
    print("  - comparison_{model}.png                 # So sánh output_steps theo model")
    print("  - grid_sample{0,1,2}.png                 # Grid tổng quan")
    print("=" * 100)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user!")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
