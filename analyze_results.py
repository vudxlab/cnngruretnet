"""
Script phân tích và so sánh kết quả từ các models với output_steps khác nhau
Thu thập metrics, tạo visualization và báo cáo
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Phân tích và so sánh kết quả training"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results_comparison',
        help='Thư mục chứa kết quả training'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='analysis',
        help='Thư mục lưu kết quả phân tích'
    )
    return parser.parse_args()


def collect_metrics(results_dir):
    """
    Thu thập metrics từ tất cả thư mục kết quả

    Args:
        results_dir: Thư mục chứa kết quả

    Returns:
        pd.DataFrame: DataFrame chứa tất cả metrics
    """
    print(f"\n📊 Đang thu thập metrics từ: {results_dir}/")

    all_metrics = []

    # Duyệt qua tất cả thư mục
    for folder_name in os.listdir(results_dir):
        folder_path = os.path.join(results_dir, folder_name)

        if not os.path.isdir(folder_path):
            continue

        # Parse folder name: {model}_out{output_step}
        try:
            parts = folder_name.split('_out')
            if len(parts) != 2:
                continue

            model = parts[0]
            output_step = int(parts[1])

        except (ValueError, IndexError):
            print(f"  ⚠️  Bỏ qua folder: {folder_name} (format không đúng)")
            continue

        # Đọc metrics.csv
        metrics_file = os.path.join(folder_path, 'metrics.csv')

        if not os.path.exists(metrics_file):
            print(f"  ⚠️  Không tìm thấy metrics.csv trong: {folder_name}")
            continue

        try:
            df = pd.read_csv(metrics_file, encoding='utf-8')

            # Thêm thông tin model và output_step
            for _, row in df.iterrows():
                all_metrics.append({
                    'model': model,
                    'output_step': output_step,
                    'dataset': row['Dataset'],
                    'rmse': row['RMSE'],
                    'mae': row['MAE'],
                    'r2': row['R2']
                })

            print(f"  ✓ {folder_name}: {len(df)} datasets")

        except Exception as e:
            print(f"  ❌ Lỗi khi đọc {folder_name}: {e}")

    # Tạo DataFrame
    metrics_df = pd.DataFrame(all_metrics)

    print(f"\n✓ Đã thu thập {len(metrics_df)} entries từ {len(metrics_df['model'].unique())} models")
    print(f"  Models: {sorted(metrics_df['model'].unique())}")
    print(f"  Output steps: {sorted(metrics_df['output_step'].unique())}")

    return metrics_df


def create_comparison_table(metrics_df, output_dir):
    """
    Tạo bảng so sánh metrics

    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục output
    """
    print(f"\n📋 Đang tạo bảng so sánh...")

    # Filter chỉ lấy Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    # Sort by model và output_step
    test_df = test_df.sort_values(['model', 'output_step'])

    # Save to CSV
    output_file = os.path.join(output_dir, 'comparison_table.csv')
    test_df.to_csv(output_file, index=False, encoding='utf-8')

    print(f"✓ Đã lưu bảng so sánh: {output_file}")

    # Print summary
    print("\n📊 BẢNG SO SÁNH (Test Set):")
    print("=" * 100)
    print(f"{'Model':<15} {'Out Steps':<12} {'RMSE':<15} {'MAE':<15} {'R²':<10}")
    print("-" * 100)

    for _, row in test_df.iterrows():
        print(f"{row['model']:<15} {row['output_step']:<12} "
              f"{row['rmse']:<15.6f} {row['mae']:<15.6f} {row['r2']:<10.4f}")

    print("=" * 100)

    return test_df


def plot_metrics_vs_output_steps(metrics_df, output_dir):
    """
    Vẽ biểu đồ metrics theo output_steps

    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục output
    """
    print(f"\n📈 Đang tạo visualization...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (18, 12)

    # Tạo 3 subplots cho 3 metrics
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    metrics_to_plot = [
        ('r2', 'R² Score', 'higher is better'),
        ('rmse', 'RMSE', 'lower is better'),
        ('mae', 'MAE', 'lower is better')
    ]

    colors = {'conv1d_gru': '#2ecc71', 'gru': '#3498db', 'conv1d': '#e74c3c'}
    markers = {'conv1d_gru': 'o', 'gru': 's', 'conv1d': '^'}

    for idx, (metric, title, note) in enumerate(metrics_to_plot):
        ax = axes[idx]

        # Plot từng model
        for model in sorted(test_df['model'].unique()):
            model_data = test_df[test_df['model'] == model].sort_values('output_step')

            ax.plot(
                model_data['output_step'],
                model_data[metric],
                marker=markers.get(model, 'o'),
                linewidth=2,
                markersize=8,
                label=model.upper(),
                color=colors.get(model, '#95a5a6')
            )

        ax.set_xlabel('Output Steps', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} vs Output Steps ({note})', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11, loc='best')
        ax.grid(True, alpha=0.3)

        # Set x-axis ticks
        output_steps = sorted(test_df['output_step'].unique())
        ax.set_xticks(output_steps)

    plt.tight_layout()

    # Save plot
    output_file = os.path.join(output_dir, 'metrics_vs_output_steps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu biểu đồ: {output_file}")


def plot_heatmap(metrics_df, output_dir):
    """
    Vẽ heatmap cho từng metric

    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục output
    """
    print(f"\n🔥 Đang tạo heatmaps...")

    # Filter Test set
    test_df = metrics_df[metrics_df['dataset'] == 'Test'].copy()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    metrics_to_plot = ['r2', 'rmse', 'mae']
    titles = ['R² Score (Higher is Better)', 'RMSE (Lower is Better)', 'MAE (Lower is Better)']
    cmaps = ['RdYlGn', 'RdYlGn_r', 'RdYlGn_r']

    for idx, (metric, title, cmap) in enumerate(zip(metrics_to_plot, titles, cmaps)):
        # Pivot table
        pivot = test_df.pivot(index='model', columns='output_step', values=metric)

        # Plot heatmap
        sns.heatmap(
            pivot,
            annot=True,
            fmt='.4f',
            cmap=cmap,
            ax=axes[idx],
            cbar_kws={'label': metric.upper()},
            linewidths=0.5
        )

        axes[idx].set_title(title, fontsize=14, fontweight='bold')
        axes[idx].set_xlabel('Output Steps', fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Model', fontsize=12, fontweight='bold')

    plt.tight_layout()

    # Save
    output_file = os.path.join(output_dir, 'heatmaps.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Đã lưu heatmaps: {output_file}")


def find_best_configurations(metrics_df, output_dir):
    """
    Tìm các cấu hình tốt nhất

    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục output
    """
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
        print(f"  Model: {result['Model']}")
        print(f"  Output Steps: {result['Output Steps']}")
        print(f"  Value: {result['Value']}")
        print(f"  RMSE: {result['RMSE']}, MAE: {result['MAE']}")
    print("=" * 100)

    return best_df


def generate_summary_report(metrics_df, output_dir):
    """
    Tạo báo cáo tổng hợp

    Args:
        metrics_df: DataFrame chứa metrics
        output_dir: Thư mục output
    """
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
    report.append(f"Models: {', '.join(sorted(test_df['model'].unique()))}")
    report.append(f"Output steps: {sorted(test_df['output_step'].unique())}")
    report.append(f"Tổng số combinations: {len(test_df)}")
    report.append("")

    # Performance summary by model
    report.append("=" * 100)
    report.append("PERFORMANCE SUMMARY BY MODEL")
    report.append("=" * 100)
    report.append("")

    for model in sorted(test_df['model'].unique()):
        model_data = test_df[test_df['model'] == model]

        report.append(f"\n{model.upper()}:")
        report.append(f"  R² Range: {model_data['r2'].min():.6f} - {model_data['r2'].max():.6f}")
        report.append(f"  RMSE Range: {model_data['rmse'].min():.6f} - {model_data['rmse'].max():.6f}")
        report.append(f"  MAE Range: {model_data['mae'].min():.6f} - {model_data['mae'].max():.6f}")

        # Best output_step for this model
        best_idx = model_data['r2'].idxmax()
        best = model_data.loc[best_idx]
        report.append(f"  Best output_step: {int(best['output_step'])} (R²={best['r2']:.6f})")

    report.append("")
    report.append("=" * 100)

    # Impact of output_steps
    report.append("IMPACT OF OUTPUT_STEPS")
    report.append("=" * 100)
    report.append("")

    for out_step in sorted(test_df['output_step'].unique()):
        step_data = test_df[test_df['output_step'] == out_step]

        report.append(f"\nOutput Steps = {out_step}:")
        report.append(f"  Average R²: {step_data['r2'].mean():.6f}")
        report.append(f"  Average RMSE: {step_data['rmse'].mean():.6f}")
        report.append(f"  Average MAE: {step_data['mae'].mean():.6f}")

        # Best model for this output_step
        best_idx = step_data['r2'].idxmax()
        best = step_data.loc[best_idx]
        report.append(f"  Best model: {best['model']} (R²={best['r2']:.6f})")

    report.append("")
    report.append("=" * 100)

    # Save report
    output_file = os.path.join(output_dir, 'summary_report.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✓ Đã lưu báo cáo: {output_file}")

    # Print to console
    print("\n" + '\n'.join(report))


def main():
    """Main function"""
    args = parse_args()

    print("=" * 100)
    print("  PHÂN TÍCH KẾT QUẢ - SO SÁNH MODELS VỚI OUTPUT_STEPS")
    print("=" * 100)

    # Kiểm tra results_dir tồn tại
    if not os.path.exists(args.results_dir):
        print(f"\n❌ Lỗi: Không tìm thấy thư mục {args.results_dir}")
        print(f"   Chạy script training trước: python compare_output_steps.py")
        sys.exit(1)

    # Tạo output directory
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n📁 Output directory: {args.output_dir}/")

    # Thu thập metrics
    metrics_df = collect_metrics(args.results_dir)

    if metrics_df.empty:
        print("\n❌ Không tìm thấy metrics nào!")
        sys.exit(1)

    # Tạo các phân tích
    create_comparison_table(metrics_df, args.output_dir)
    plot_metrics_vs_output_steps(metrics_df, args.output_dir)
    plot_heatmap(metrics_df, args.output_dir)
    find_best_configurations(metrics_df, args.output_dir)
    generate_summary_report(metrics_df, args.output_dir)

    print("\n" + "=" * 100)
    print("✅ HOÀN THÀNH PHÂN TÍCH!")
    print("=" * 100)
    print(f"\n📁 Kết quả phân tích lưu tại: {args.output_dir}/")
    print("\nCác files đã tạo:")
    print("  - comparison_table.csv       # Bảng so sánh đầy đủ")
    print("  - metrics_vs_output_steps.png # Biểu đồ metrics theo output_steps")
    print("  - heatmaps.png                # Heatmaps cho các metrics")
    print("  - best_configurations.csv     # Các cấu hình tốt nhất")
    print("  - summary_report.txt          # Báo cáo tổng hợp")
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
