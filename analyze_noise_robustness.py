"""
Script phân tích robustness của model với các noise_factor khác nhau
Cấu trúc: results/noise/{noise_factor}/cnn_resnet_gru/metrics.csv
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Phân tích robustness của model với các noise levels"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/noise',
        help='Thư mục chứa kết quả noise experiments (default: results/noise/)'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='cnn_resnet_gru',
        help='Model cần phân tích (default: cnn_resnet_gru)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis/noise',
        help='Thư mục lưu kết quả phân tích (default: analysis/noise/)'
    )
    return parser.parse_args()


def collect_noise_metrics(results_dir, model_name):
    """
    Thu thập metrics từ các noise_factor folders

    Args:
        results_dir: Thư mục chứa kết quả (e.g., results/noise)
        model_name: Tên model (e.g., cnn_resnet_gru)

    Returns:
        pd.DataFrame: DataFrame chứa metrics với noise_factor
    """
    print(f"\n📊 Đang thu thập metrics từ: {results_dir}/")
    print(f"   Model: {model_name}")
    print(f"   Cấu trúc: {results_dir}/{{noise_factor}}/{model_name}/metrics.csv")

    all_metrics = []

    # Kiểm tra thư mục tồn tại
    if not os.path.exists(results_dir):
        print(f"\n❌ Không tìm thấy thư mục: {results_dir}")
        return pd.DataFrame()

    # Duyệt qua các noise_factor folders
    for noise_folder in sorted(os.listdir(results_dir)):
        noise_path = os.path.join(results_dir, noise_folder)

        if not os.path.isdir(noise_path):
            continue

        # Parse noise_factor từ folder name
        try:
            noise_factor = float(noise_folder)
        except ValueError:
            print(f"  ⚠️  Bỏ qua folder: {noise_folder} (không phải số)")
            continue

        # Đường dẫn đến metrics.csv
        metrics_file = os.path.join(noise_path, model_name, 'metrics.csv')

        if not os.path.exists(metrics_file):
            print(f"  ⚠️  Không tìm thấy: {noise_folder}/{model_name}/metrics.csv")
            continue

        try:
            # Đọc metrics
            df = pd.read_csv(metrics_file)

            # Thêm noise_factor column
            for _, row in df.iterrows():
                all_metrics.append({
                    'noise_factor': noise_factor,
                    'dataset': row['Dataset'],
                    'rmse': row['RMSE'],
                    'mae': row['MAE'],
                    'r2': row['R2']
                })

            print(f"  ✓ Noise={noise_factor}: {len(df)} datasets")

        except Exception as e:
            print(f"  ❌ Lỗi khi đọc {noise_folder}: {e}")

    # Tạo DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    if metrics_df.empty:
        print("\n❌ Không thu thập được metrics nào!")
        return metrics_df

    print(f"\n✓ Đã thu thập {len(metrics_df)} entries")
    print(f"  Noise factors: {sorted(metrics_df['noise_factor'].unique())}")
    print(f"  Datasets: {sorted(metrics_df['dataset'].unique())}")

    return metrics_df


def create_comparison_table(metrics_df, output_dir, model_name):
    """Tạo bảng so sánh metrics theo noise_factor"""
    print(f"\n📋 Đang tạo bảng so sánh...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    if test_df.empty:
        print("  ⚠️  Không có dữ liệu Test set")
        return

    # Pivot table
    table_data = []
    for noise in sorted(test_df['noise_factor'].unique()):
        row_data = test_df[test_df['noise_factor'] == noise].iloc[0]
        table_data.append({
            'Noise Factor': noise,
            'RMSE': row_data['rmse'],
            'MAE': row_data['mae'],
            'R²': row_data['r2']
        })

    comparison_df = pd.DataFrame(table_data)

    # Save
    output_file = os.path.join(output_dir, 'noise_comparison_table.csv')
    comparison_df.to_csv(output_file, index=False)
    print(f"✓ Đã lưu bảng so sánh: {output_file}")

    # Print table
    print("\n" + "=" * 80)
    print(f"📊 BẢNG SO SÁNH - {model_name.upper().replace('_', '-')} (Test Set)")
    print("=" * 80)
    print(f"{'Noise Factor':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
    print("-" * 80)
    for _, row in comparison_df.iterrows():
        print(f"{row['Noise Factor']:<15.2f} {row['RMSE']:<15.6f} {row['MAE']:<15.6f} {row['R²']:<15.6f}")
    print("=" * 80)

    return comparison_df


def plot_metrics_vs_noise(metrics_df, output_dir, model_name):
    """Vẽ biểu đồ metrics theo noise_factor"""
    print(f"\n📈 Đang tạo line plots...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()
    test_df = test_df.sort_values('noise_factor')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ('r2', 'R² Score', 'Higher is Better'),
        ('rmse', 'RMSE', 'Lower is Better'),
        ('mae', 'MAE', 'Lower is Better')
    ]

    colors = ['#2ecc71', '#e74c3c', '#3498db']

    for idx, (metric, title, note) in enumerate(metrics):
        ax = axes[idx]

        # Plot line
        ax.plot(test_df['noise_factor'], test_df[metric],
               marker='o', linewidth=2, markersize=8, color=colors[idx],
               label=f'{title}')

        # Fill area under curve
        ax.fill_between(test_df['noise_factor'], test_df[metric],
                        alpha=0.2, color=colors[idx])

        # Formatting
        ax.set_xlabel('Noise Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} vs Noise Level\n({note})',
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=9, loc='best')

        # Add value labels
        for x, y in zip(test_df['noise_factor'], test_df[metric]):
            ax.annotate(f'{y:.4f}', (x, y),
                       textcoords="offset points", xytext=(0,5),
                       ha='center', fontsize=8)

    plt.suptitle(f'Noise Robustness Analysis - {model_name.upper().replace("_", "-")}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'metrics_vs_noise.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu biểu đồ: {output_file}")


def plot_bar_comparison(metrics_df, output_dir, model_name):
    """Vẽ biểu đồ cột so sánh"""
    print(f"\n📊 Đang tạo bar charts...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()
    test_df = test_df.sort_values('noise_factor')

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    metrics = [
        ('r2', 'R² Score', '#2ecc71'),
        ('rmse', 'RMSE', '#e74c3c'),
        ('mae', 'MAE', '#3498db')
    ]

    x_pos = np.arange(len(test_df))

    for idx, (metric, title, color) in enumerate(metrics):
        ax = axes[idx]

        # Bar plot
        bars = ax.bar(x_pos, test_df[metric], color=color, alpha=0.7,
                     edgecolor='black', linewidth=1.5)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, test_df[metric])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Formatting
        ax.set_xlabel('Noise Factor', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(f'{title} by Noise Level', fontsize=12, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{nf:.2f}' for nf in test_df['noise_factor']])
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.suptitle(f'Performance Degradation Analysis - {model_name.upper().replace("_", "-")}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'bar_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu biểu đồ: {output_file}")


def plot_heatmap(metrics_df, output_dir, model_name):
    """Vẽ heatmap cho tất cả metrics"""
    print(f"\n🔥 Đang tạo heatmap...")

    # Pivot data
    pivot_data = []
    for dataset in ['Train', 'Validation', 'Test']:
        dataset_df = metrics_df[metrics_df['dataset'] == dataset].copy()
        dataset_df = dataset_df.sort_values('noise_factor')

        for metric in ['r2', 'rmse', 'mae']:
            for _, row in dataset_df.iterrows():
                pivot_data.append({
                    'Metric': f"{metric.upper()} ({dataset})",
                    'Noise Factor': row['noise_factor'],
                    'Value': row[metric]
                })

    pivot_df = pd.DataFrame(pivot_data)
    heatmap_data = pivot_df.pivot(index='Metric', columns='Noise Factor', values='Value')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 6))

    sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn_r',
               linewidths=1, linecolor='black', cbar_kws={'label': 'Metric Value'},
               ax=ax)

    ax.set_title(f'Performance Heatmap - {model_name.upper().replace("_", "-")}',
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Noise Factor', fontsize=12, fontweight='bold')
    ax.set_ylabel('Metric', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu heatmap: {output_file}")


def calculate_degradation(metrics_df, output_dir, model_name):
    """Tính toán performance degradation"""
    print(f"\n📉 Đang tính toán performance degradation...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()
    test_df = test_df.sort_values('noise_factor')

    # Baseline (lowest noise)
    baseline = test_df.iloc[0]
    baseline_noise = baseline['noise_factor']

    degradation_data = []

    for _, row in test_df.iterrows():
        noise = row['noise_factor']

        # Calculate degradation
        r2_drop = baseline['r2'] - row['r2']
        rmse_increase = row['rmse'] - baseline['rmse']
        mae_increase = row['mae'] - baseline['mae']

        # Calculate percentage
        r2_drop_pct = (r2_drop / baseline['r2']) * 100 if baseline['r2'] != 0 else 0
        rmse_increase_pct = (rmse_increase / baseline['rmse']) * 100 if baseline['rmse'] != 0 else 0
        mae_increase_pct = (mae_increase / baseline['mae']) * 100 if baseline['mae'] != 0 else 0

        degradation_data.append({
            'Noise Factor': noise,
            'R² Drop': r2_drop,
            'R² Drop (%)': r2_drop_pct,
            'RMSE Increase': rmse_increase,
            'RMSE Increase (%)': rmse_increase_pct,
            'MAE Increase': mae_increase,
            'MAE Increase (%)': mae_increase_pct
        })

    degradation_df = pd.DataFrame(degradation_data)

    # Save
    output_file = os.path.join(output_dir, 'degradation_analysis.csv')
    degradation_df.to_csv(output_file, index=False)
    print(f"✓ Đã lưu degradation analysis: {output_file}")

    # Print summary
    print("\n" + "=" * 100)
    print(f"📉 PERFORMANCE DEGRADATION (Baseline: Noise={baseline_noise:.2f})")
    print("=" * 100)
    print(f"{'Noise':<10} {'R² Drop':<15} {'R² Drop %':<15} {'RMSE Inc':<15} {'RMSE Inc %':<15} {'MAE Inc':<15} {'MAE Inc %':<15}")
    print("-" * 100)
    for _, row in degradation_df.iterrows():
        print(f"{row['Noise Factor']:<10.2f} {row['R² Drop']:<15.6f} {row['R² Drop (%)']:<15.2f} "
              f"{row['RMSE Increase']:<15.6f} {row['RMSE Increase (%)']:<15.2f} "
              f"{row['MAE Increase']:<15.6f} {row['MAE Increase (%)']:<15.2f}")
    print("=" * 100)

    return degradation_df


def plot_overlay_predictions(results_dir, model_name, output_dir, num_samples=5):
    """
    Vẽ overlay comparison của predictions với các noise levels khác nhau

    Args:
        results_dir: Thư mục chứa kết quả (e.g., results/noise)
        model_name: Tên model (e.g., cnn_resnet_gru)
        output_dir: Thư mục output
        num_samples: Số samples để vẽ
    """
    print(f"\n📊 Đang tạo overlay predictions comparison...")

    try:
        import tensorflow as tf
    except ImportError:
        print("  ⚠️  TensorFlow không available, bỏ qua overlay predictions")
        return

    # Collect predictions từ các noise folders
    predictions_dict = {}
    noise_factors = []

    for noise_folder in sorted(os.listdir(results_dir)):
        noise_path = os.path.join(results_dir, noise_folder)

        if not os.path.isdir(noise_path):
            continue

        try:
            noise_factor = float(noise_folder)
        except ValueError:
            continue

        # Load model và predictions
        model_path = os.path.join(noise_path, model_name)
        model_file = os.path.join(model_path, 'model_saved.keras')

        if not os.path.exists(model_file):
            print(f"  ⚠️  Không tìm thấy model: {noise_folder}/{model_name}/model_saved.keras")
            continue

        try:
            # Load model
            keras_model = tf.keras.models.load_model(model_file)

            # Load predictions từ folder nếu có
            predictions_folder = os.path.join(model_path, 'predictions')
            y_test_file = os.path.join(predictions_folder, 'y_test.npy')
            y_pred_file = os.path.join(predictions_folder, 'y_pred.npy')

            if os.path.exists(y_test_file) and os.path.exists(y_pred_file):
                y_test = np.load(y_test_file)
                y_pred = np.load(y_pred_file)

                predictions_dict[noise_factor] = {
                    'y_test': y_test,
                    'y_pred': y_pred
                }
                noise_factors.append(noise_factor)
                print(f"  ✓ Loaded noise={noise_factor}: {len(y_test)} samples")
            else:
                print(f"  ⚠️  Không tìm thấy predictions files cho noise={noise_factor}")

        except Exception as e:
            print(f"  ❌ Lỗi khi load predictions cho noise={noise_folder}: {e}")

    if not predictions_dict:
        print("  ⚠️  Không có predictions data để vẽ overlay")
        return

    noise_factors = sorted(noise_factors)

    # Chọn samples để vẽ (lấy samples có prediction tốt nhất từ noise thấp nhất)
    baseline_noise = noise_factors[0]
    y_test_baseline = predictions_dict[baseline_noise]['y_test']
    y_pred_baseline = predictions_dict[baseline_noise]['y_pred']

    # Tính MSE cho mỗi sample
    mse_per_sample = []
    for i in range(len(y_test_baseline)):
        mse = np.mean((y_test_baseline[i] - y_pred_baseline[i]) ** 2)
        mse_per_sample.append((i, mse))

    # Sort và lấy top num_samples
    mse_per_sample.sort(key=lambda x: x[1])
    best_indices = [idx for idx, _ in mse_per_sample[:num_samples]]

    print(f"  ✓ Đã chọn {num_samples} samples tốt nhất từ noise={baseline_noise}")

    # Define colors cho từng noise level
    colors_palette = ['#2ecc71', '#3498db', '#f39c12', '#e74c3c', '#9b59b6', '#1abc9c']
    noise_colors = {nf: colors_palette[i % len(colors_palette)]
                    for i, nf in enumerate(noise_factors)}

    # Vẽ overlay plots
    fig, axes = plt.subplots(num_samples, 1, figsize=(14, 4*num_samples))

    if num_samples == 1:
        axes = [axes]

    for sample_idx, sample_index in enumerate(best_indices):
        ax = axes[sample_idx]

        # Plot actual (chỉ cần 1 lần từ bất kỳ noise nào)
        y_true = predictions_dict[baseline_noise]['y_test'][sample_index]
        timesteps = range(len(y_true))

        ax.plot(timesteps, y_true, 'o-', linewidth=2.5, markersize=6,
               label='Actual', color='black', alpha=0.8, zorder=10)

        # Plot predictions từ tất cả noise levels
        for noise_factor in noise_factors:
            y_pred = predictions_dict[noise_factor]['y_pred'][sample_index]

            ax.plot(timesteps, y_pred, 's--', linewidth=2, markersize=5,
                   label=f'Predicted (Noise={noise_factor:.2f})',
                   color=noise_colors[noise_factor], alpha=0.7)

        # Formatting
        ax.set_xlabel('Time Step', fontsize=11, fontweight='bold')
        ax.set_ylabel('Value', fontsize=11, fontweight='bold')
        ax.set_title(f'Sample {sample_idx+1} - Noise Robustness Comparison',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best', framealpha=0.9, ncol=2)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

    plt.suptitle(f'Prediction Overlay - {model_name.upper().replace("_", "-")} across Noise Levels',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'overlay_predictions.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu overlay predictions: {output_file}")


def create_summary_report(metrics_df, degradation_df, output_dir, model_name):
    """Tạo báo cáo tổng hợp"""
    print(f"\n📝 Đang tạo báo cáo tổng hợp...")

    report = []
    report.append("=" * 100)
    report.append("NOISE ROBUSTNESS ANALYSIS REPORT")
    report.append("=" * 100)
    report.append(f"\nModel: {model_name.upper().replace('_', '-')}")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Test set summary
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()
    test_df = test_df.sort_values('noise_factor')

    report.append("\n" + "=" * 100)
    report.append("TEST SET PERFORMANCE SUMMARY")
    report.append("=" * 100)
    report.append(f"\n{'Noise Factor':<15} {'R² Score':<15} {'RMSE':<15} {'MAE':<15}")
    report.append("-" * 60)
    for _, row in test_df.iterrows():
        report.append(f"{row['noise_factor']:<15.2f} {row['r2']:<15.6f} {row['rmse']:<15.6f} {row['mae']:<15.6f}")

    # Best/Worst noise levels
    best_r2_row = test_df.loc[test_df['r2'].idxmax()]
    worst_r2_row = test_df.loc[test_df['r2'].idxmin()]

    report.append("\n" + "=" * 100)
    report.append("KEY FINDINGS")
    report.append("=" * 100)
    report.append(f"\n✓ Best Performance: Noise Factor = {best_r2_row['noise_factor']:.2f}")
    report.append(f"  - R² = {best_r2_row['r2']:.6f}")
    report.append(f"  - RMSE = {best_r2_row['rmse']:.6f}")
    report.append(f"  - MAE = {best_r2_row['mae']:.6f}")

    report.append(f"\n✗ Worst Performance: Noise Factor = {worst_r2_row['noise_factor']:.2f}")
    report.append(f"  - R² = {worst_r2_row['r2']:.6f}")
    report.append(f"  - RMSE = {worst_r2_row['rmse']:.6f}")
    report.append(f"  - MAE = {worst_r2_row['mae']:.6f}")

    # Performance degradation
    max_degradation = degradation_df.loc[degradation_df['R² Drop (%)'].idxmax()]

    report.append(f"\n📉 Maximum R² Degradation: {max_degradation['R² Drop (%)']:.2f}% at Noise Factor = {max_degradation['Noise Factor']:.2f}")
    report.append(f"📈 Maximum RMSE Increase: {max_degradation['RMSE Increase (%)']:.2f}% at Noise Factor = {max_degradation['Noise Factor']:.2f}")

    # Robustness assessment
    avg_r2_drop = degradation_df['R² Drop (%)'].mean()
    report.append(f"\n📊 Average R² Drop: {avg_r2_drop:.2f}%")

    if avg_r2_drop < 5:
        robustness = "EXCELLENT"
    elif avg_r2_drop < 10:
        robustness = "GOOD"
    elif avg_r2_drop < 20:
        robustness = "MODERATE"
    else:
        robustness = "POOR"

    report.append(f"\n🎯 Robustness Assessment: {robustness}")
    report.append("=" * 100)

    # Save report
    report_text = "\n".join(report)
    output_file = os.path.join(output_dir, 'noise_robustness_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"✓ Đã lưu báo cáo: {output_file}")

    # Print to console
    print("\n" + report_text)


def main():
    """Main function"""
    args = parse_args()

    print("=" * 100)
    print("  NOISE ROBUSTNESS ANALYSIS")
    print("=" * 100)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir}/")

    # Collect metrics
    metrics_df = collect_noise_metrics(args.results_dir, args.model)

    if metrics_df.empty:
        print("\n❌ Không có dữ liệu để phân tích!")
        sys.exit(1)

    # Create analyses
    print("\n" + "=" * 100)
    print("  ĐANG TẠO PHÂN TÍCH")
    print("=" * 100)

    # 1. Comparison table
    comparison_df = create_comparison_table(metrics_df, args.output_dir, args.model)

    # 2. Line plots
    plot_metrics_vs_noise(metrics_df, args.output_dir, args.model)

    # 3. Bar charts
    plot_bar_comparison(metrics_df, args.output_dir, args.model)

    # 4. Heatmap
    plot_heatmap(metrics_df, args.output_dir, args.model)

    # 5. Degradation analysis
    degradation_df = calculate_degradation(metrics_df, args.output_dir, args.model)

    # 6. Overlay predictions comparison
    plot_overlay_predictions(args.results_dir, args.model, args.output_dir, num_samples=5)

    # 7. Summary report
    create_summary_report(metrics_df, degradation_df, args.output_dir, args.model)

    print("\n" + "=" * 100)
    print("✅ HOÀN THÀNH PHÂN TÍCH!")
    print("=" * 100)
    print(f"\n📁 Kết quả lưu tại: {args.output_dir}/")
    print("\nCác files đã tạo:")
    print("  ✓ noise_comparison_table.csv      # Bảng so sánh đầy đủ")
    print("  ✓ metrics_vs_noise.png             # Line plots cho các metrics")
    print("  ✓ bar_comparison.png               # Bar charts so sánh")
    print("  ✓ heatmap.png                      # Heatmap tổng quan")
    print("  ✓ degradation_analysis.csv         # Phân tích performance degradation")
    print("  ✓ overlay_predictions.png          # Overlay predictions với các noise levels")
    print("  ✓ noise_robustness_report.txt      # Báo cáo tổng hợp")
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
