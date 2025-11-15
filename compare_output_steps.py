"""
Script để train và so sánh tất cả models với các output_steps khác nhau
So sánh predictions và metrics của từng output_step
"""

import os
import sys
import subprocess
from datetime import datetime


# Cấu hình
MODELS = ['conv1d_gru', 'gru', 'conv1d']  # 3 Deep Learning models
OUTPUT_STEPS = [5, 10, 15, 20, 30, 40]     # Tất cả output steps
EPOCHS = 1000
BATCH_SIZE = 64
BASE_OUTPUT_DIR = "results_comparison"


def print_header(title, width=80):
    """In header đẹp"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width + "\n")


def train_combination(model, output_step, base_dir, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """
    Train một combination của model + output_step

    Args:
        model: Tên model ('conv1d_gru', 'gru', 'conv1d')
        output_step: Số output steps (5, 10, 15, 20, 30, 40)
        base_dir: Thư mục output base
        epochs: Số epochs
        batch_size: Batch size

    Returns:
        dict: Kết quả training
    """
    output_dir = os.path.join(base_dir, f"{model}_out{output_step}")

    print_header(f"TRAINING: {model.upper()} - OUTPUT_STEPS={output_step}")
    print(f"Model: {model}")
    print(f"Output steps: {output_step}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {epochs}, Batch size: {batch_size}\n")

    # Build command
    cmd = [
        sys.executable,
        'main.py',
        '--models', model,
        '--output_steps', str(output_step),
        '--output_dir', output_dir,
        '--epochs', str(epochs),
        '--batch_size', str(batch_size),
    ]

    # Run training
    start_time = datetime.now()

    try:
        result = subprocess.run(cmd, check=True, capture_output=False)

        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()

        print(f"\n✅ {model} (out={output_step}) trained successfully!")
        print(f"   Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        return {
            'model': model,
            'output_step': output_step,
            'status': 'success',
            'time': elapsed,
            'output_dir': output_dir
        }

    except subprocess.CalledProcessError as e:
        print(f"\n❌ {model} (out={output_step}) FAILED!")
        print(f"   Error: {e}")

        return {
            'model': model,
            'output_step': output_step,
            'status': 'failed',
            'error': str(e)
        }
    except KeyboardInterrupt:
        print(f"\n⚠️  {model} (out={output_step}) INTERRUPTED by user!")
        raise
    except Exception as e:
        print(f"\n❌ {model} (out={output_step}) FAILED with unexpected error!")
        print(f"   Error: {e}")

        return {
            'model': model,
            'output_step': output_step,
            'status': 'failed',
            'error': str(e)
        }


def main():
    """Main function"""
    print_header("SO SÁNH MODELS VỚI CÁC OUTPUT_STEPS", width=80)

    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nCấu hình:")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Output steps: {OUTPUT_STEPS}")
    print(f"  Total combinations: {len(MODELS)} x {len(OUTPUT_STEPS)} = {len(MODELS) * len(OUTPUT_STEPS)}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Output directory: {BASE_OUTPUT_DIR}/")

    # Tạo base output directory
    os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

    results = []
    total_combinations = len(MODELS) * len(OUTPUT_STEPS)
    current = 0

    # Train tất cả combinations
    for model in MODELS:
        for output_step in OUTPUT_STEPS:
            current += 1

            print("\n" + "=" * 80)
            print(f"  COMBINATION {current}/{total_combinations}")
            print(f"  Model: {model.upper()} | Output Steps: {output_step}")
            print("=" * 80)

            result = train_combination(
                model=model,
                output_step=output_step,
                base_dir=BASE_OUTPUT_DIR,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            results.append(result)

    # Print summary
    print_header("TRAINING SUMMARY", width=80)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print(f"Total combinations: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}\n")

    # Summary table
    print(f"{'Model':<15} {'Output Steps':<15} {'Status':<12} {'Time (min)':<15} {'Output Dir'}")
    print("-" * 100)

    for r in results:
        status = "✅ SUCCESS" if r['status'] == 'success' else "❌ FAILED"
        time_str = f"{r.get('time', 0)/60:.1f}" if r['status'] == 'success' else "N/A"
        output_dir = r.get('output_dir', 'N/A')

        print(f"{r['model']:<15} {r['output_step']:<15} {status:<12} {time_str:<15} {output_dir}")

    print("=" * 100)

    if failed == 0:
        print("\n🎉 Tất cả combinations đã train thành công!")
        print(f"\n📊 Tiếp theo, chạy script phân tích:")
        print(f"   python analyze_results.py --results_dir {BASE_OUTPUT_DIR}")
    else:
        print(f"\n⚠️  {failed} combination(s) thất bại. Kiểm tra logs phía trên.")

    print(f"\n📁 Kết quả lưu tại: {BASE_OUTPUT_DIR}/")
    print("=" * 80)

    return failed == 0


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user!")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
