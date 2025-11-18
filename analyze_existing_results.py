"""
Script phân tích kết quả từ folder structure hiện có:
results/
├── 5/           (output_steps=5)
│   ├── conv1d_gru/
│   ├── gru/
│   └── conv1d/
├── 10/          (output_steps=10)
│   ├── conv1d_gru/
│   ├── gru/
│   └── conv1d/
└── ...
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils import get_model_display_name, parse_model_type_from_path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Phân tích kết quả từ folder structure hiện có"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='Result_no_aug',
        help='Thư mục chứa kết quả (default: results/)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis_no_aug',
        help='Thư mục lưu kết quả phân tích (default: analysis/)'
    )
    parser.add_argument(
        '--plot_predictions',
        action='store_true',
        help='Tạo prediction comparison plots (mất thêm thời gian)'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=5,
        help='Số samples cho prediction plots (default: 5)'
    )
    return parser.parse_args()


def collect_metrics_from_structure(results_dir):
    """
    Thu thập metrics từ folder structure:
    - Cấu trúc 1: results/{output_step}/{model}/ (ví dụ: results/5/cnn/)
    - Cấu trúc 2: results/{model}/ (ví dụ: results/revision/5/cnn/)

    Args:
        results_dir: Thư mục chứa kết quả

    Returns:
        pd.DataFrame: DataFrame chứa tất cả metrics
    """
    print(f"\n📊 Đang thu thập metrics từ: {results_dir}/")

    all_metrics = []

    # Try to detect output_step from path
    output_step = None
    path_parts = Path(results_dir).parts
    for part in reversed(path_parts):
        try:
            output_step = int(part)
            print(f"   ✓ Detected output_step = {output_step} từ path")
            break
        except ValueError:
            continue

    # Duyệt qua các folders trong results_dir
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Check nếu folder này chứa metrics.csv → đây là model folder
        metrics_file = os.path.join(folder_path, 'metrics.csv')

        if os.path.exists(metrics_file):
            # Đây là model folder
            model_path = folder_path
            model_folder = folder_name

            # Parse model type và num_gru_layers từ folder name
            model_type, num_gru_layers = parse_model_type_from_path(model_path)

            # Convert sang display name
            model_display = get_model_display_name(model_type, num_gru_layers)

            try:
                df = pd.read_csv(metrics_file, encoding='utf-8')

                # Thêm thông tin model và output_step
                for _, row in df.iterrows():
                    all_metrics.append({
                        'model': model_display,  # Sử dụng display name
                        'model_type': model_type,  # Giữ lại model type gốc
                        'output_step': output_step if output_step else 5,  # Default 5 nếu không detect được
                        'dataset': row['Dataset'],
                        'rmse': row['RMSE'],
                        'mae': row['MAE'],
                        'r2': row['R2']
                    })

                print(f"    ✓ {model_display}: {len(df)} datasets")

            except Exception as e:
                print(f"    ❌ Lỗi khi đọc {model_folder}: {e}")
        else:
            # Không có metrics.csv, có thể là output_step folder
            # Try to check subfolders
            try:
                step_num = int(folder_name)
                # Đây là output_step folder, duyệt subfolder
                print(f"\n  📂 Output steps = {step_num}")

                for model_subfolder in os.listdir(folder_path):
                    model_subpath = os.path.join(folder_path, model_subfolder)

                    if not os.path.isdir(model_subpath):
                        continue

                    metrics_subfile = os.path.join(model_subpath, 'metrics.csv')

                    if not os.path.exists(metrics_subfile):
                        continue

                    # Parse model
                    model_type, num_gru_layers = parse_model_type_from_path(model_subpath)
                    model_display = get_model_display_name(model_type, num_gru_layers)

                    try:
                        df = pd.read_csv(metrics_subfile, encoding='utf-8')

                        for _, row in df.iterrows():
                            all_metrics.append({
                                'model': model_display,
                                'model_type': model_type,
                                'output_step': step_num,
                                'dataset': row['Dataset'],
                                'rmse': row['RMSE'],
                                'mae': row['MAE'],
                                'r2': row['R2']
                            })

                        print(f"    ✓ {model_display}: {len(df)} datasets")

                    except Exception as e:
                        print(f"    ❌ Lỗi khi đọc {model_subfolder}: {e}")

            except ValueError:
                # Không phải số, skip
                pass

    # Tạo DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    if metrics_df.empty:
        print("\n❌ Không thu thập được metrics nào!")
        return metrics_df

    print(f"\n✓ Đã thu thập {len(metrics_df)} entries")
    print(f"  Models: {sorted(metrics_df['model'].unique())}")
    print(f"  Output steps: {sorted(metrics_df['output_step'].unique())}")
    print(f"  Datasets: {sorted(metrics_df['dataset'].unique())}")

    return metrics_df


def create_comparison_table(metrics_df, output_dir):
    """Tạo bảng so sánh metrics"""
    print(f"\n📋 Đang tạo bảng so sánh...")

    # Filter chỉ lấy Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    # Sort by output_step và model
    test_df = test_df.sort_values(['output_step', 'model'])

    # Save to CSV
    output_file = os.path.join(output_dir, 'comparison_table.csv')
    test_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"✓ Đã lưu bảng so sánh: {output_file}")

    # Print summary
    print("\n📊 BẢNG SO SÁNH (Test Set):")
    print("=" * 100)
    print(f"{'Out Steps':<12} {'Model':<15} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
    print("-" * 100)

    for _, row in test_df.iterrows():
        print(f"{row['output_step']:<12} {row['model']:<15} "
              f"{row['rmse']:<15.6f} {row['mae']:<15.6f} {row['r2']:<10.4f}")

    print("=" * 100)

    return test_df


def plot_metrics_vs_output_steps(metrics_df, output_dir):
    """Vẽ biểu đồ metrics theo output_steps"""
    print(f"\n📈 Đang tạo visualization...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (18, 12)

    # Tạo 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    metrics_to_plot = [
        ('r2', 'R² Score', 'higher is better'),
        ('rmse', 'RMSE', 'lower is better'),
        ('mae', 'MAE', 'lower is better')
    ]

    # Define colors và markers cho tất cả models (sử dụng display names)
    colors = {
        'CNN+ResNet+GRU': '#2ecc71',  # Green
        'CNN+GRU': '#3498db',  # Blue
        'CNN+ResNet': '#e74c3c',  # Red
        'CNN+ResNet+GRU+BN': '#9b59b6',  # Purple
        'CNN': '#f39c12',  # Orange
        'GRU': '#1abc9c',  # Teal
        'Linear Regression': '#95a5a6',  # Gray
        'XGBoost': '#e67e22',  # Dark orange
        'LightGBM': '#16a085',  # Dark teal
    }

    markers = {
        'CNN+ResNet+GRU': 'o',
        'CNN+GRU': 's',
        'CNN+ResNet': '^',
        'CNN+ResNet+GRU+BN': 'D',
        'CNN': 'v',
        'GRU': 'p',
        'Linear Regression': 'x',
        'XGBoost': '+',
        'LightGBM': '*',
    }

    # Tạo colors/markers động cho models không có trong dict
    default_colors = ['#34495e', '#7f8c8d', '#c0392b', '#8e44ad', '#27ae60', '#2980b9']
    default_markers = ['H', '8', 'P', 'd', '<', '>']

    for idx, (metric, title, note) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot từng model
        model_list = sorted(test_df['model'].unique())
        for i, model in enumerate(model_list):
            model_data = test_df[test_df['model'] == model].sort_values('output_step')

            # Get color và marker (với fallback nếu không có trong dict)
            color = colors.get(model, default_colors[i % len(default_colors)])
            marker = markers.get(model, default_markers[i % len(default_markers)])

            ax.plot(
                model_data['output_step'],
                model_data[metric],
                marker=marker,
                linewidth=2.5,
                markersize=10,
                label=model,  # model đã là display name
                color=color
            )

        ax.set_xlabel('Output Steps', fontsize=13, fontweight='bold')
        ax.set_ylabel(title, fontsize=13, fontweight='bold')
        ax.set_title(f'{title} vs Output Steps ({note})', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=12, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Set x-axis ticks
        output_steps = sorted(test_df['output_step'].unique())
        ax.set_xticks(output_steps)

        # Format y-axis
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.4f}'))

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, 'metrics_vs_output_steps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu biểu đồ: {output_file}")


def plot_heatmap(metrics_df, output_dir):
    """Vẽ heatmap cho từng metric"""
    print(f"\n🔥 Đang tạo heatmaps...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    metrics_to_plot = ['r2', 'rmse', 'mae']
    titles = ['R² Score\n(Higher is Better)', 'RMSE\n(Lower is Better)', 'MAE\n(Lower is Better)']
    cmaps = ['RdYlGn', 'RdYlGn_r', 'RdYlGn_r']

    for idx, (metric, title, cmap) in enumerate(zip(metrics_to_plot, titles, cmaps)):
        # Pivot table
        pivot = test_df.pivot(index='model', columns='output_step', values=metric)

        # Model names are already display names, no need to rename

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap=cmap,
            ax=axes[idx],
            cbar_kws={'label': metric.upper()},
            linewidths=1,
            linecolor='white',
            annot_kws={'fontsize': 10}
        )

        axes[idx].set_title(title, fontsize=14, fontweight='bold', pad=15)
        axes[idx].set_xlabel('Output Steps', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Model', fontsize=12, fontweight='bold')

        # Rotate labels
        axes[idx].set_xticklabels(axes[idx].get_xticklabels(), rotation=0)
        axes[idx].set_yticklabels(axes[idx].get_yticklabels(), rotation=0)

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'heatmaps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu heatmaps: {output_file}")


def find_best_configurations(metrics_df, output_dir):
    """Tìm các cấu hình tốt nhất"""
    print(f"\n🏆 Đang tìm best configurations...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    results = []

    # Best R² (highest)
    best_r2 = test_df.loc[test_df['r2'].idxmax()]
    results.append({
        'Metric': 'Best R²',
        'Model': best_r2['model'],
        'Output Steps': int(best_r2['output_step']),
        'Value': f"{best_r2['r2']:.6f}",
        'RMSE': f"{best_r2['rmse']:.6f}",
        'MAE': f"{best_r2['mae']:.6f}"
    })

    # Best RMSE (lowest)
    best_rmse = test_df.loc[test_df['rmse'].idxmin()]
    results.append({
        'Metric': 'Best RMSE',
        'Model': best_rmse['model'],
        'Output Steps': int(best_rmse['output_step']),
        'Value': f"{best_rmse['rmse']:.6f}",
        'RMSE': f"{best_rmse['rmse']:.6f}",
        'MAE': f"{best_rmse['mae']:.6f}"
    })

    # Best MAE (lowest)
    best_mae = test_df.loc[test_df['mae'].idxmin()]
    results.append({
        'Metric': 'Best MAE',
        'Model': best_mae['model'],
        'Output Steps': int(best_mae['output_step']),
        'Value': f"{best_mae['mae']:.6f}",
        'RMSE': f"{best_mae['rmse']:.6f}",
        'MAE': f"{best_mae['mae']:.6f}"
    })

    # Save to CSV
    best_df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, 'best_configurations.csv')
    best_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"✓ Đã lưu best configurations: {output_file}")

    # Print
    print("\n🏆 BEST CONFIGURATIONS:")
    print("=" * 100)
    for result in results:
        print(f"\n{result['Metric']}:")
        print(f"  Model: {result['Model'].upper()}")
        print(f"  Output Steps: {result['Output Steps']}")
        print(f"  Value: {result['Value']}")
        print(f"  RMSE: {result['RMSE']}, MAE: {result['MAE']}")
    print("=" * 100)

    return best_df


def generate_summary_report(metrics_df, output_dir):
    """Tạo báo cáo tổng hợp"""
    print(f"\n📝 Đang tạo báo cáo tổng hợp...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    report = []
    report.append("=" * 100)
    report.append("BÁO CÁO SO SÁNH MODELS VỚI CÁC OUTPUT_STEPS")
    report.append("=" * 100)
    report.append("")

    # General stats
    report.append(f"Tổng số models: {len(test_df['model'].unique())}")
    report.append(f"Models: {', '.join([m.upper() for m in sorted(test_df['model'].unique())])}")
    report.append(f"Output steps: {sorted(test_df['output_step'].unique())}")
    report.append(f"Tổng số combinations: {len(test_df)}")
    report.append("")

    # Performance summary by model
    report.append("=" * 100)
    report.append("PERFORMANCE SUMMARY BY MODEL (Test Set)")
    report.append("=" * 100)
    report.append("")

    for model in sorted(test_df['model'].unique()):
        model_data = test_df[test_df['model'] == model].sort_values('output_step')

        report.append(f"\n{model.upper().replace('_', '-')}:")
        report.append(f"{'':4}R² Range: {model_data['r2'].min():.6f} → {model_data['r2'].max():.6f}")
        report.append(f"{'':4}RMSE Range: {model_data['rmse'].min():.6f} → {model_data['rmse'].max():.6f}")
        report.append(f"{'':4}MAE Range: {model_data['mae'].min():.6f} → {model_data['mae'].max():.6f}")

        # Best output_step for this model
        best_idx = model_data['r2'].idxmax()
        best = model_data.loc[best_idx]
        worst_idx = model_data['r2'].idxmin()
        worst = model_data.loc[worst_idx]

        report.append(f"{'':4}Best output_step: {int(best['output_step'])} "
                     f"(R²={best['r2']:.6f}, RMSE={best['rmse']:.6f})")
        report.append(f"{'':4}Worst output_step: {int(worst['output_step'])} "
                     f"(R²={worst['r2']:.6f}, RMSE={worst['rmse']:.6f})")

        # Performance degradation
        perf_degradation = ((best['r2'] - worst['r2']) / best['r2']) * 100
        report.append(f"{'':4}Performance degradation: {perf_degradation:.2f}%")

    report.append("")
    report.append("=" * 100)

    # Impact of output_steps
    report.append("IMPACT OF OUTPUT_STEPS ON AVERAGE PERFORMANCE")
    report.append("=" * 100)
    report.append("")

    for out_step in sorted(test_df['output_step'].unique()):
        step_data = test_df[test_df['output_step'] == out_step]

        report.append(f"\nOutput Steps = {out_step}:")
        report.append(f"{'':4}Average R²: {step_data['r2'].mean():.6f} ± {step_data['r2'].std():.6f}")
        report.append(f"{'':4}Average RMSE: {step_data['rmse'].mean():.6f} ± {step_data['rmse'].std():.6f}")
        report.append(f"{'':4}Average MAE: {step_data['mae'].mean():.6f} ± {step_data['mae'].std():.6f}")

        # Best model for this output_step
        best_idx = step_data['r2'].idxmax()
        best = step_data.loc[best_idx]
        report.append(f"{'':4}Best model: {best['model'].upper()} (R²={best['r2']:.6f})")

    report.append("")
    report.append("=" * 100)

    # Save report
    output_file = os.path.join(output_dir, 'summary_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✓ Đã lưu báo cáo: {output_file}")

    # Print to console
    print("\n" + '\n'.join(report))


def plot_training_curves_comparison(results_dir, output_dir):
    """
    Vẽ biểu đồ so sánh training curves (loss) của 3 models cho mỗi output_step

    Args:
        results_dir: Thư mục chứa kết quả
        output_dir: Thư mục output
    """
    import pickle

    print(f"\n📈 Đang tạo training curves comparison...")

    # Detect models và output_steps
    models = []
    output_steps = []

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            out_step = int(folder)
            output_steps.append(out_step)

            for model_folder in os.listdir(folder_path):
                if os.path.isdir(os.path.join(folder_path, model_folder)):
                    if model_folder not in models:
                        models.append(model_folder)
        except ValueError:
            continue

    models = sorted(models)
    output_steps = sorted(output_steps)

    if not models or not output_steps:
        print("\n⚠️  Không tìm thấy dữ liệu!")
        return

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

    # Tạo folder output
    curves_dir = os.path.join(output_dir, 'training_curves')
    os.makedirs(curves_dir, exist_ok=True)

    # Vẽ cho từng output_step
    for out_step in output_steps:
        print(f"\n  📊 Output step = {out_step}")

        # Load history cho tất cả models
        histories = {}

        for model in models:
            history_file = os.path.join(results_dir, str(out_step), model, 'history_saved.pkl')

            if not os.path.exists(history_file):
                print(f"    ⚠️  Không tìm thấy history: {model}")
                continue

            try:
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                histories[model] = history
                print(f"    ✓ Loaded {model}: {len(history.get('loss', []))} epochs")
            except Exception as e:
                print(f"    ❌ Lỗi load {model}: {e}")

        if not histories:
            print(f"    ⚠️  Không có history nào cho output_step={out_step}")
            continue

        # Vẽ biểu đồ (2 subplots: Train Loss & Val Loss)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Training Loss
        ax1 = axes[0]
        for model, history in histories.items():
            if 'loss' in history:
                epochs = range(1, len(history['loss']) + 1)
                model_display_name = model_names.get(model, model.upper().replace('_', '-'))
                ax1.plot(epochs, history['loss'],
                        linewidth=2, label=model_display_name,
                        color=colors.get(model, '#95a5a6'), alpha=0.8)

        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
        ax1.set_title(f'Training Loss - Output Steps = {out_step}',
                     fontsize=14, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_yscale('log')  # Log scale cho loss

        # Subplot 2: Validation Loss
        ax2 = axes[1]
        for model, history in histories.items():
            if 'val_loss' in history:
                epochs = range(1, len(history['val_loss']) + 1)
                model_display_name = model_names.get(model, model.upper().replace('_', '-'))
                ax2.plot(epochs, history['val_loss'],
                        linewidth=2, label=model_display_name,
                        color=colors.get(model, '#95a5a6'), alpha=0.8)

        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
        ax2.set_title(f'Validation Loss - Output Steps = {out_step}',
                     fontsize=14, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_yscale('log')  # Log scale cho loss

        plt.tight_layout()

        # Save
        output_file = os.path.join(curves_dir, f'training_curves_out{out_step}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"    ✓ Đã lưu: {output_file}")

    print(f"\n✓ Training curves comparison đã lưu tại: {curves_dir}/")


def generate_prediction_comparisons(results_dir, output_dir, num_samples=5):
    """
    Tạo prediction comparison visualizations

    Args:
        results_dir: Thư mục chứa kết quả
        output_dir: Thư mục output
        num_samples: Số samples để vẽ
    """
    # Import here to avoid dependency if not needed
    try:
        from plot_prediction_comparison import (
            plot_overlay_comparison
            # Các hàm khác đã bị disable: chỉ giữ overlay
            # plot_comparison_by_output_step,
            # plot_comparison_by_model,
            # plot_all_combinations_grid
        )
    except ImportError:
        print("\n⚠️  Không thể import plot_prediction_comparison")
        return

    print("\n" + "=" * 100)
    print("  TẠO PREDICTION COMPARISON VISUALIZATIONS")
    print("=" * 100)

    # Detect models và output_steps
    models = []
    output_steps = []

    for folder in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        try:
            out_step = int(folder)
            output_steps.append(out_step)

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
        print("\n⚠️  Không tìm thấy dữ liệu cho predictions!")
        return

    # 1. Overlay comparison (CẢ 3 models trên cùng subplot - KHUYÊN DÙNG)
    print("\n1️⃣  Overlay Comparison (3 models on same plot - RECOMMENDED):")
    for out_step in output_steps:
        plot_overlay_comparison(results_dir, out_step, models,
                               output_dir, num_samples=num_samples)

    # 2. Comparison by output_step (separate subplots)
    # DISABLED: Không cần thiết, chỉ giữ overlay
    # print("\n2️⃣  Comparison by Output Step (separate subplots):")
    # for out_step in output_steps:
    #     plot_comparison_by_output_step(results_dir, out_step, models,
    #                                   output_dir, num_samples=num_samples)

    # 3. Comparison by model
    # DISABLED: Không cần thiết, chỉ giữ overlay
    # print("\n3️⃣  Comparison by Model:")
    # for model in models:
    #     plot_comparison_by_model(results_dir, model, output_steps,
    #                             output_dir, num_samples=num_samples)

    # 4. Grid overview
    # DISABLED: Không cần thiết, chỉ giữ overlay
    # print("\n4️⃣  Overview Grid:")
    # for sample_idx in range(min(3, num_samples)):
    #     plot_all_combinations_grid(results_dir, models, output_steps,
    #                                output_dir, sample_idx=sample_idx)

    print(f"\n✅ Prediction comparisons đã lưu tại: {output_dir}/predictions_comparison/")


def main():
    """Main function"""
    args = parse_args()

    print("=" * 100)
    print("  PHÂN TÍCH KẾT QUẢ - SO SÁNH MODELS VỚI OUTPUT_STEPS")
    print("=" * 100)

    # Kiểm tra results_dir tồn tại
    if not os.path.exists(args.results_dir):
        print(f"\n❌ Lỗi: Không tìm thấy thư mục {args.results_dir}")
        sys.exit(1)

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir}/")

    # Thu thập metrics
    metrics_df = collect_metrics_from_structure(args.results_dir)

    if metrics_df.empty:
        print("\n❌ Không tìm thấy metrics nào!")
        sys.exit(1)

    # Tạo các phân tích metrics
    create_comparison_table(metrics_df, args.output_dir)
    plot_metrics_vs_output_steps(metrics_df, args.output_dir)
    plot_heatmap(metrics_df, args.output_dir)
    find_best_configurations(metrics_df, args.output_dir)
    generate_summary_report(metrics_df, args.output_dir)

    # Tạo training curves comparison
    plot_training_curves_comparison(args.results_dir, args.output_dir)

    # Tạo prediction comparisons nếu được yêu cầu
    if args.plot_predictions:
        generate_prediction_comparisons(args.results_dir, args.output_dir,
                                       num_samples=args.num_samples)

    print("\n" + "=" * 100)
    print("✅ HOÀN THÀNH PHÂN TÍCH!")
    print("=" * 100)
    print(f"\n📁 Kết quả phân tích lưu tại: {args.output_dir}/")
    print("\nCác files đã tạo:")
    print("  ✓ comparison_table.csv       # Bảng so sánh đầy đủ")
    print("  ✓ metrics_vs_output_steps.png # Biểu đồ metrics theo output_steps")
    print("  ✓ heatmaps.png                # Heatmaps cho các metrics")
    print("  ✓ best_configurations.csv     # Các cấu hình tốt nhất")
    print("  ✓ summary_report.txt          # Báo cáo tổng hợp")
    print("\n  📈 Training Curves:")
    print("  ✓ training_curves/training_curves_out*.png  # So sánh loss curves của 3 models")

    if args.plot_predictions:
        print("\n  📊 Prediction Comparisons:")
        print("  🌟 predictions_comparison/overlay_out*.png        # Overlay 3 models trên cùng 1 biểu đồ")

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
