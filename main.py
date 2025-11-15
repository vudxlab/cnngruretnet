"""
Main Entry Point
Chạy toàn bộ pipeline: Load data -> Preprocess -> Build model -> Train -> Evaluate
Hỗ trợ train nhiều models trong một lần chạy
"""

import argparse
import sys
import os
from datetime import datetime

from config import Config
from data_loader import load_vibration_data
from data_preprocessing import preprocess_data
from model import create_model
from trainer import train_model
from evaluator import evaluate_model
from visualization import Visualizer
from utils import (set_random_seed, create_output_directory, print_separator,
                  ModelSaver, count_model_parameters)


def parse_arguments():
    """
    Parse command line arguments
    """
    parser = argparse.ArgumentParser(
        description="Time Series Forecasting với Conv1D-GRU (Fixed Data Leakage)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Train một model
  python main.py --models conv1d_gru

  # Train nhiều models
  python main.py --models conv1d_gru gru conv1d

  # Train tất cả deep learning models
  python main.py --models conv1d_gru gru conv1d --epochs 500

  # Train cả baseline models
  python main.py --models linear xgboost lightgbm
        """
    )

    # Data parameters
    parser.add_argument("--mat_file", type=str, default=None,
                       help="Đường dẫn file .mat (mặc định: lấy từ Config)")
    parser.add_argument("--sensor_idx", type=int, default=Config.SENSOR_IDX,
                       help="Index sensor (0-7)")

    # Model parameters - ĐÃ ĐỔI TỪ model_type SANG models
    parser.add_argument("--models", type=str, nargs='+', default=['conv1d_gru'],
                       choices=['conv1d_gru', 'conv1d', 'gru', 'linear', 'xgboost', 'lightgbm'],
                       help="Loại model(s) cần train (có thể chọn nhiều)")

    # Training parameters
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS,
                       help="Số epochs")
    parser.add_argument("--batch_size", type=int, default=Config.BATCH_SIZE,
                       help="Batch size")
    parser.add_argument("--patience", type=int, default=Config.EARLY_STOPPING_PATIENCE,
                       help="Early stopping patience")

    # Data augmentation
    parser.add_argument("--add_noise", action="store_true", default=Config.ADD_NOISE,
                       help="Có thêm noise không")
    parser.add_argument("--no_noise", action="store_true",
                       help="KHÔNG thêm noise")

    # Output
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Thư mục output base (mỗi model sẽ có subfolder riêng)")

    # Seed
    parser.add_argument("--seed", type=int, default=Config.SEED,
                       help="Random seed")

    # Modes
    parser.add_argument("--skip_training", action="store_true",
                       help="Bỏ qua training (chỉ load model)")
    parser.add_argument("--skip_evaluation", action="store_true",
                       help="Bỏ qua evaluation")
    parser.add_argument("--skip_visualization", action="store_true",
                       help="Bỏ qua visualization")

    args = parser.parse_args()

    # Handle no_noise flag
    if args.no_noise:
        args.add_noise = False

    return args


def train_single_model(model_type, data_dict, args, base_output_dir):
    """
    Train một model duy nhất

    Args:
        model_type: Loại model ('conv1d_gru', 'gru', 'conv1d', etc.)
        data_dict: Dictionary chứa X_train, y_train, X_val, y_val, X_test, y_test, preprocessor
        args: Command line arguments
        base_output_dir: Thư mục output base

    Returns:
        dict: Kết quả training (metrics, paths, status)
    """
    # Tạo output directory cho model này
    model_output_dir = os.path.join(base_output_dir, model_type)

    # Update Config cho model này
    Config.OUTPUT_DIR = model_output_dir

    print_separator(f"MODEL: {model_type.upper()}", width=70)

    # Create output directory
    create_output_directory(model_output_dir)

    # Unpack data
    X_train = data_dict['X_train']
    y_train = data_dict['y_train']
    X_val = data_dict['X_val']
    y_val = data_dict['y_val']
    X_test = data_dict['X_test']
    y_test = data_dict['y_test']
    preprocessor = data_dict['preprocessor']

    # Save scaler
    scaler_path = Config.get_output_path(Config.SCALER_FILE)
    preprocessor.save_scaler(scaler_path)

    # ==================== BUILD MODEL ====================
    print(f"\n[{model_type}] Building model...")

    model = create_model(
        model_type=model_type,
        input_steps=Config.INPUT_STEPS,
        output_steps=Config.OUTPUT_STEPS,
        n_features=Config.N_FEATURES,
        compile_model=True
    )

    # Print model info
    params = count_model_parameters(model)
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,}")

    # ==================== TRAIN MODEL ====================
    if not args.skip_training:
        print(f"\n[{model_type}] Training...")

        start_time = datetime.now()

        trainer, history = train_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=args.epochs,
            batch_size=args.batch_size
        )

        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()

        # Save model
        model_path = Config.get_output_path(Config.MODEL_FILE)
        ModelSaver.save_model(model, model_path)

        # Save history
        history_path = Config.get_output_path(Config.HISTORY_FILE)
        ModelSaver.save_history(history, history_path)

        # Save training time
        time_path = Config.get_output_path(Config.TIME_LOG_FILE)
        ModelSaver.save_training_time(trainer.get_training_time(), time_path)

        # Visualize training history
        if not args.skip_visualization:
            Visualizer.plot_training_history(history, save_path=model_output_dir)

    else:
        print(f"\n[{model_type}] Loading model (skip training)...")
        model_path = Config.get_output_path(Config.MODEL_FILE)
        model = ModelSaver.load_model(model_path)
        training_time = 0

    # ==================== EVALUATE MODEL ====================
    metrics = None
    if not args.skip_evaluation:
        print(f"\n[{model_type}] Evaluating...")

        evaluator, metrics = evaluate_model(
            model, preprocessor,
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            save_metrics=True
        )

        # Visualize metrics comparison
        if not args.skip_visualization:
            metrics_plot_path = Config.get_output_path("metrics_comparison.png")
            Visualizer.plot_metrics_comparison(metrics, save_path=metrics_plot_path)

    # ==================== VISUALIZE PREDICTIONS ====================
    if not args.skip_visualization:
        print(f"\n[{model_type}] Generating prediction plots...")

        # Predict trên test set
        y_pred = model.predict(X_test[:10], verbose=0)

        Visualizer.plot_predictions(
            X_test[:10], y_test[:10], y_pred,
            num_samples=5,
            preprocessor=preprocessor,
            save_dir=Config.get_output_path("predictions")
        )

    # ==================== RETURN RESULTS ====================
    result = {
        'model_type': model_type,
        'output_dir': model_output_dir,
        'training_time': training_time,
        'metrics': metrics,
        'status': 'success'
    }

    print(f"\n✅ [{model_type}] Hoàn thành!")
    if metrics:
        test_metrics = metrics.get('test', {})
        print(f"   Test R²: {test_metrics.get('r2', 0):.4f}")
        print(f"   Test RMSE: {test_metrics.get('rmse', 0):.6f}")
    print(f"   Lưu tại: {model_output_dir}")

    return result


def main():
    """
    Main function - chạy toàn bộ pipeline
    """
    # Parse arguments
    args = parse_arguments()

    # Update Config từ args (global settings)
    Config.SENSOR_IDX = args.sensor_idx
    Config.EPOCHS = args.epochs
    Config.BATCH_SIZE = args.batch_size
    Config.EARLY_STOPPING_PATIENCE = args.patience
    Config.SEED = args.seed

    # Print header
    print_separator("TIME SERIES FORECASTING PROJECT", width=70)
    print("Fixed Data Leakage - Modular Architecture")
    print_separator(width=70)

    print(f"\nModels to train: {', '.join(args.models)}")
    print(f"Base output directory: {args.output_dir}")

    # Set random seed
    set_random_seed(args.seed)

    # Print configuration
    print("\n")
    Config.print_config()

    # ==================== STEP 1: LOAD DATA ====================
    print_separator("STEP 1: LOAD DATA", width=70)

    data_sensor = load_vibration_data(
        mat_file_path=args.mat_file,
        sensor_idx=args.sensor_idx
    )

    # ==================== STEP 2: PREPROCESS DATA ====================
    print_separator("STEP 2: PREPROCESS DATA", width=70)

    data_dict = preprocess_data(data_sensor, add_noise=args.add_noise)

    print(f"\n✓ Data shapes:")
    print(f"  X_train: {data_dict['X_train'].shape}")
    print(f"  y_train: {data_dict['y_train'].shape}")
    print(f"  X_val: {data_dict['X_val'].shape}")
    print(f"  X_test: {data_dict['X_test'].shape}")

    # ==================== STEP 3-6: TRAIN MODELS ====================
    print_separator("STEP 3-6: BUILD, TRAIN & EVALUATE MODELS", width=70)

    results = []

    for i, model_type in enumerate(args.models, 1):
        print(f"\n{'='*70}")
        print(f"  TRAINING MODEL {i}/{len(args.models)}: {model_type.upper()}")
        print(f"{'='*70}\n")

        try:
            result = train_single_model(
                model_type=model_type,
                data_dict=data_dict,
                args=args,
                base_output_dir=args.output_dir
            )
            results.append(result)

        except Exception as e:
            print(f"\n❌ [{model_type}] FAILED: {str(e)}")
            import traceback
            traceback.print_exc()

            results.append({
                'model_type': model_type,
                'status': 'failed',
                'error': str(e)
            })

    # ==================== SUMMARY ====================
    print_separator("TRAINING SUMMARY", width=70)

    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\nTotal models: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}\n")

    print(f"{'Model':<20} {'Status':<12} {'Test R²':<12} {'Output Directory'}")
    print("-" * 80)

    for r in results:
        status = "✅ SUCCESS" if r['status'] == 'success' else "❌ FAILED"
        r2_score = ""

        if r['status'] == 'success' and r.get('metrics'):
            test_metrics = r['metrics'].get('test', {})
            r2_score = f"{test_metrics.get('r2', 0):.4f}"

        output = r.get('output_dir', 'N/A')
        print(f"{r['model_type']:<20} {status:<12} {r2_score:<12} {output}")

    print_separator(width=70)

    if failed == 0:
        print("\n🎉 Tất cả models đã train thành công!")
    else:
        print(f"\n⚠️  {failed} model(s) thất bại. Kiểm tra logs phía trên.")

    print(f"\nKết quả lưu tại: {args.output_dir}/")
    print_separator(width=70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nĐã dừng bởi người dùng (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nLỗi: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
