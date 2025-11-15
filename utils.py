"""
Utilities Module
Các hàm tiện ích: lưu/load model, history, logging, etc.
"""

import os
import pickle
import csv
import numpy as np
import tensorflow as tf
from config import Config


class ModelSaver:
    """
    Class quản lý việc lưu/load model và artifacts
    """

    @staticmethod
    def save_model(model, filepath=None):
        """
        Lưu Keras model

        Args:
            model: Keras model
            filepath: Đường dẫn file (nếu None, dùng từ Config)
        """
        filepath = filepath or Config.get_output_path(Config.MODEL_FILE)
        model.save(filepath)
        print(f"✓ Đã lưu model tại: {filepath}")

    @staticmethod
    def load_model(filepath=None):
        """
        Load Keras model

        Args:
            filepath: Đường dẫn file

        Returns:
            model: Keras model
        """
        filepath = filepath or Config.get_output_path(Config.MODEL_FILE)
        model = tf.keras.models.load_model(filepath)
        print(f"✓ Đã load model từ: {filepath}")
        return model

    @staticmethod
    def save_history(history, filepath=None):
        """
        Lưu training history

        Args:
            history: Keras History object hoặc history.history dict
            filepath: Đường dẫn file
        """
        filepath = filepath or Config.get_output_path(Config.HISTORY_FILE)

        # Lấy history dict nếu là History object
        if hasattr(history, 'history'):
            history_dict = history.history
        else:
            history_dict = history

        with open(filepath, 'wb') as f:
            pickle.dump(history_dict, f)

        print(f"✓ Đã lưu history tại: {filepath}")

    @staticmethod
    def load_history(filepath=None):
        """
        Load training history

        Args:
            filepath: Đường dẫn file

        Returns:
            history_dict: Dictionary chứa history
        """
        filepath = filepath or Config.get_output_path(Config.HISTORY_FILE)

        with open(filepath, 'rb') as f:
            history_dict = pickle.load(f)

        print(f"✓ Đã load history từ: {filepath}")
        return history_dict

    @staticmethod
    def save_training_time(elapsed_time, filepath=None):
        """
        Lưu thời gian training vào CSV

        Args:
            elapsed_time: Thời gian (giây)
            filepath: Đường dẫn file
        """
        filepath = filepath or Config.get_output_path(Config.TIME_LOG_FILE)

        with open(filepath, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["Thời gian train (giây)"])
            writer.writerow([round(elapsed_time, 2)])

        print(f"✓ Đã lưu training time tại: {filepath}")


class Logger:
    """
    Simple logger để ghi log
    """

    def __init__(self, log_file=None):
        self.log_file = log_file

    def log(self, message, print_console=True):
        """
        Ghi log

        Args:
            message: Message cần log
            print_console: Có in ra console không
        """
        if print_console:
            print(message)

        if self.log_file:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')


def set_random_seed(seed=None):
    """
    Set random seed cho reproducibility

    Args:
        seed: Random seed (nếu None, lấy từ Config)
    """
    seed = seed or Config.SEED

    np.random.seed(seed)
    tf.random.set_seed(seed)

    print(f"✓ Set random seed = {seed}")


def create_output_directory(output_dir=None):
    """
    Tạo thư mục output

    Args:
        output_dir: Đường dẫn thư mục (nếu None, dùng từ Config)
    """
    output_dir = output_dir or Config.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"✓ Đã tạo/kiểm tra thư mục output: {output_dir}")


def print_separator(title=None, width=60, char="="):
    """
    In dòng separator

    Args:
        title: Tiêu đề (nếu có)
        width: Độ rộng
        char: Ký tự dùng để vẽ
    """
    if title:
        padding = (width - len(title) - 2) // 2
        print(char * padding + f" {title} " + char * padding)
    else:
        print(char * width)


def get_model_summary_string(model):
    """
    Lấy model summary dưới dạng string

    Args:
        model: Keras model

    Returns:
        str: Model summary
    """
    import io
    from contextlib import redirect_stdout

    f = io.StringIO()
    with redirect_stdout(f):
        model.summary()
    return f.getvalue()


def count_model_parameters(model):
    """
    Đếm số parameters trong model

    Args:
        model: Keras model

    Returns:
        dict: {'total': X, 'trainable': Y, 'non_trainable': Z}
    """
    trainable = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable = sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights])
    total = trainable + non_trainable

    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': non_trainable
    }


if __name__ == "__main__":
    # Test utils
    print("Testing Utils...")

    # Test set random seed
    set_random_seed(42)

    # Test create output directory
    create_output_directory("test_output")

    # Test separator
    print_separator("TEST TITLE")
    print_separator()

    # Test model saver (với dummy model)
    from model import create_model

    model = create_model('conv1d_gru')

    # Save model
    ModelSaver.save_model(model, "test_output/test_model.keras")

    # Load model
    loaded_model = ModelSaver.load_model("test_output/test_model.keras")

    # Count parameters
    params = count_model_parameters(model)
    print(f"\nModel parameters: {params}")

    print("\n✓ Utils test completed!")
