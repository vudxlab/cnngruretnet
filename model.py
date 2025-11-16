"""
Model Architecture Module
Định nghĩa các kiến trúc mô hình cho time series forecasting
"""

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv1D, Add, GRU, LSTM, Dense, Dropout, BatchNormalization, Flatten
)
from config import Config


class Conv1DGRUModel:
    """
    Conv1D-GRU Model với Skip Connection
    Kiến trúc hybrid kết hợp:
    - Conv1D: Trích xuất features từ time series
    - Skip Connection: Giúp gradient flow tốt hơn (ResNet-inspired)
    - GRU: Học temporal dependencies
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        """
        Khởi tạo model

        Args:
            input_steps: Số timesteps đầu vào
            output_steps: Số timesteps đầu ra
            n_features: Số features
        """
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """
        Xây dựng model architecture

        Returns:
            model: Keras Model
        """
        print("\nXây dựng Conv1D-GRU Model...")

        # Input layer
        input_layer = Input(shape=(self.input_steps, self.n_features), name='input')

        # Conv1D layer
        conv_out = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            activation=Config.CONV_ACTIVATION,
            padding=Config.CONV_PADDING,
            strides=1,
            name='conv1d'
        )(input_layer)

        # Resize input để match conv_out filters (cho skip connection)
        input_resized = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=1,
            activation='linear',
            padding='same',
            name='input_resize'
        )(input_layer)

        # Skip Connection (Add)
        x = Add(name='skip_connection')([conv_out, input_resized])

        # GRU layers
        x = GRU(
            units=Config.GRU_UNITS_1,
            activation=Config.GRU_ACTIVATION,
            recurrent_activation=Config.GRU_RECURRENT_ACTIVATION,
            return_sequences=True,
            name='gru_1'
        )(x)

        x = GRU(
            units=Config.GRU_UNITS_2,
            activation=Config.GRU_ACTIVATION,
            recurrent_activation=Config.GRU_RECURRENT_ACTIVATION,
            return_sequences=True,
            name='gru_2'
        )(x)

        x = GRU(
            units=Config.GRU_UNITS_3,
            activation=Config.GRU_ACTIVATION,
            recurrent_activation=Config.GRU_RECURRENT_ACTIVATION,
            return_sequences=False,  # Chỉ lấy output cuối cùng
            name='gru_3'
        )(x)

        # Dense layers với capacity lớn hơn và regularization (giống CNN-ResNet model)
        x = Dense(128, activation='relu', name='dense_1')(x)
        x = Dropout(0.2, name='dropout_1')(x)
        x = Dense(64, activation='relu', name='dense_2')(x)

        # Output layer
        output_layer = Dense(units=self.output_steps, name='output')(x)

        # Build model
        self.model = Model(inputs=input_layer, outputs=output_layer, name='Conv1D_GRU')

        print("✓ Model đã được xây dựng")
        return self.model

    def compile(self, optimizer=None, loss=None, metrics=None):
        """
        Compile model

        Args:
            optimizer: Optimizer (mặc định từ Config)
            loss: Loss function (mặc định từ Config)
            metrics: List metrics (mặc định từ Config)
        """
        if self.model is None:
            raise ValueError("Model chưa được build! Gọi build() trước.")

        optimizer = optimizer or Config.OPTIMIZER
        loss = loss or Config.LOSS
        metrics = metrics or Config.METRICS

        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

        print(f"✓ Model đã compile với optimizer={optimizer}, loss={loss}")

    def summary(self):
        """In summary của model"""
        if self.model is None:
            raise ValueError("Model chưa được build!")

        return self.model.summary()

    def get_model(self):
        """Trả về Keras model"""
        return self.model


class Conv1DModel:
    """
    Conv1D Model thuần (baseline)
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """Xây dựng Conv1D model"""
        print("\nXây dựng Conv1D Model...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # Conv1D layers với BatchNorm và Dropout
        x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)

        # Flatten và Dense
        from tensorflow.keras.layers import Flatten
        x = Flatten()(x)
        output_layer = Dense(self.output_steps)(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='Conv1D')
        print("✓ Conv1D Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        """Compile model"""
        if self.model is None:
            raise ValueError("Model chưa được build!")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


class GRUModel:
    """
    GRU Model thuần (baseline)
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """Xây dựng GRU model"""
        print("\nXây dựng GRU Model...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # 3 GRU layers
        x = GRU(128, activation='tanh', return_sequences=True)(input_layer)
        x = GRU(64, activation='tanh', return_sequences=True)(x)
        x = GRU(32, activation='tanh', return_sequences=False)(x)

        output_layer = Dense(self.output_steps)(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='GRU')
        print("✓ GRU Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        """Compile model"""
        if self.model is None:
            raise ValueError("Model chưa được build!")

        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


# ==================== ABLATION STUDY MODELS ====================

class CNNGRUNoResidual:
    """
    CNN+GRU Model KHÔNG CÓ Skip Connection
    Ablation: Để test tác động của residual connection
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """Xây dựng CNN+GRU model without residual"""
        print("\nXây dựng CNN+GRU Model (No Residual)...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # Conv1D layer (KHÔNG có skip connection)
        conv_out = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            activation=Config.CONV_ACTIVATION,
            padding=Config.CONV_PADDING,
            name='conv1d'
        )(input_layer)

        # GRU layers (giống như model gốc)
        x = GRU(Config.GRU_UNITS_1, activation='tanh', return_sequences=True, name='gru_1')(conv_out)
        x = GRU(Config.GRU_UNITS_2, activation='tanh', return_sequences=True, name='gru_2')(x)
        x = GRU(Config.GRU_UNITS_3, activation='tanh', return_sequences=False, name='gru_3')(x)

        output_layer = Dense(self.output_steps, name='output')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='CNN_GRU_NoResidual')
        print("✓ CNN+GRU (No Residual) Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        if self.model is None:
            raise ValueError("Model chưa được build!")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


class CNNResNetNoGRU:
    """
    CNN+ResNet Model KHÔNG CÓ GRU
    Ablation: Để test tác động của GRU layers
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """Xây dựng CNN+ResNet model without GRU"""
        print("\nXây dựng CNN+ResNet Model (No GRU)...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # Conv1D layer với skip connection
        conv_out = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            activation=Config.CONV_ACTIVATION,
            padding=Config.CONV_PADDING,
            name='conv1d'
        )(input_layer)

        # Resize input để match conv_out filters
        input_resized = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=1,
            activation='linear',
            padding='same',
            name='input_resize'
        )(input_layer)

        # Skip Connection
        x = Add(name='skip_connection')([conv_out, input_resized])

        # Flatten và Dense (KHÔNG có GRU)
        from tensorflow.keras.layers import Flatten
        x = Flatten()(x)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        output_layer = Dense(self.output_steps, name='output')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='CNN_ResNet_NoGRU')
        print("✓ CNN+ResNet (No GRU) Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        if self.model is None:
            raise ValueError("Model chưa được build!")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


class CNNGRUWithBatchNorm:
    """
    CNN+GRU+ResNet với BatchNorm và Dropout
    Ablation: Để test tác động của regularization
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.model = None

    def build(self):
        """Xây dựng CNN+GRU model với BatchNorm và Dropout"""
        print("\nXây dựng CNN+GRU+ResNet Model (with BatchNorm/Dropout)...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # Conv1D layer với BatchNorm và Dropout
        conv_out = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            activation=Config.CONV_ACTIVATION,
            padding=Config.CONV_PADDING,
            name='conv1d'
        )(input_layer)
        conv_out = BatchNormalization()(conv_out)
        conv_out = Dropout(0.2)(conv_out)

        # Resize input
        input_resized = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=1,
            activation='linear',
            padding='same',
            name='input_resize'
        )(input_layer)

        # Skip Connection
        x = Add(name='skip_connection')([conv_out, input_resized])

        # GRU layers với Dropout
        x = GRU(Config.GRU_UNITS_1, activation='tanh', return_sequences=True, name='gru_1')(x)
        x = Dropout(0.2)(x)
        x = GRU(Config.GRU_UNITS_2, activation='tanh', return_sequences=True, name='gru_2')(x)
        x = Dropout(0.2)(x)
        x = GRU(Config.GRU_UNITS_3, activation='tanh', return_sequences=False, name='gru_3')(x)

        output_layer = Dense(self.output_steps, name='output')(x)

        self.model = Model(inputs=input_layer, outputs=output_layer, name='CNN_GRU_BatchNorm')
        print("✓ CNN+GRU (BatchNorm/Dropout) Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        if self.model is None:
            raise ValueError("Model chưa được build!")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


class CNNGRUVariableDepth:
    """
    CNN+GRU+ResNet với số lượng GRU layers có thể config
    Ablation: Để test tác động của model depth
    """

    def __init__(self, input_steps=None, output_steps=None, n_features=None, num_gru_layers=3):
        self.input_steps = input_steps or Config.INPUT_STEPS
        self.output_steps = output_steps or Config.OUTPUT_STEPS
        self.n_features = n_features or Config.N_FEATURES
        self.num_gru_layers = num_gru_layers  # 1, 2, 3, hoặc 4
        self.model = None

    def build(self):
        """Xây dựng CNN+GRU model với số lượng GRU layers tùy chỉnh"""
        print(f"\nXây dựng CNN+GRU+ResNet Model ({self.num_gru_layers} GRU layers)...")

        input_layer = Input(shape=(self.input_steps, self.n_features))

        # Conv1D layer với skip connection
        conv_out = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=Config.CONV_KERNEL_SIZE,
            activation=Config.CONV_ACTIVATION,
            padding=Config.CONV_PADDING,
            name='conv1d'
        )(input_layer)

        input_resized = Conv1D(
            filters=Config.CONV_FILTERS,
            kernel_size=1,
            activation='linear',
            padding='same',
            name='input_resize'
        )(input_layer)

        x = Add(name='skip_connection')([conv_out, input_resized])

        # GRU layers (số lượng tùy chỉnh)
        gru_units = [128, 64, 32, 16]  # Units cho tối đa 4 layers

        for i in range(self.num_gru_layers):
            return_sequences = (i < self.num_gru_layers - 1)  # Layer cuối cùng return_sequences=False
            x = GRU(
                units=gru_units[i],
                activation='tanh',
                return_sequences=return_sequences,
                name=f'gru_{i+1}'
            )(x)

        output_layer = Dense(self.output_steps, name='output')(x)

        self.model = Model(
            inputs=input_layer,
            outputs=output_layer,
            name=f'CNN_GRU_ResNet_{self.num_gru_layers}L'
        )
        print(f"✓ CNN+GRU+ResNet ({self.num_gru_layers} layers) Model đã được xây dựng")
        return self.model

    def compile(self, optimizer='adam', loss='mse', metrics=['mae']):
        if self.model is None:
            raise ValueError("Model chưa được build!")
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"✓ Model đã compile")

    def get_model(self):
        return self.model


# ==================== FACTORY FUNCTION ====================

def create_model(model_type='conv1d_gru', input_steps=None, output_steps=None,
                n_features=None, compile_model=True, num_gru_layers=3):
    """
    Factory function để tạo model theo loại

    Args:
        model_type: Model type string
        input_steps, output_steps, n_features: Model parameters
        compile_model: Có compile model luôn không
        num_gru_layers: Số lượng GRU layers (cho CNNGRUVariableDepth)

    Returns:
        model: Keras Model đã build (và compile nếu compile_model=True)
                hoặc Baseline model instance
    """
    # Deep learning models
    dl_models = {
        # Main models
        'cnn_resnet_gru': Conv1DGRUModel,
        'cnn': Conv1DModel,
        'gru': GRUModel,
        # Ablation models
        'cnn_gru': CNNGRUNoResidual,
        'cnn_resnet': CNNResNetNoGRU,
        'cnn_resnet_gru_bn': CNNGRUWithBatchNorm,
        'cnn_resnet_gru_var': CNNGRUVariableDepth,
    }

    # Baseline models
    baseline_types = ['linear', 'xgboost', 'lightgbm', 'lr', 'xgb', 'lgbm']

    model_type_lower = model_type.lower()

    # Nếu là deep learning model
    if model_type_lower in dl_models:
        # Tạo instance (special case cho CNNGRUVariableDepth)
        if model_type_lower == 'cnn_resnet_gru_var':
            model_builder = dl_models[model_type_lower](
                input_steps=input_steps,
                output_steps=output_steps,
                n_features=n_features,
                num_gru_layers=num_gru_layers
            )
        else:
            model_builder = dl_models[model_type_lower](
                input_steps=input_steps,
                output_steps=output_steps,
                n_features=n_features
            )

        # Build
        model_builder.build()

        # Compile nếu cần
        if compile_model:
            model_builder.compile()

        # Get Keras model
        keras_model = model_builder.get_model()

        # Summary
        print("\n" + "=" * 60)
        keras_model.summary()
        print("=" * 60)

        return keras_model

    # Nếu là baseline model
    elif model_type_lower in baseline_types:
        from baseline_models import create_baseline_model
        print(f"\n⚠️  Đang tạo baseline model '{model_type}'")
        print("Lưu ý: Baseline models không dùng TensorFlow, không có compile/summary")
        return create_baseline_model(model_type_lower)

    else:
        all_types = list(dl_models.keys()) + baseline_types
        raise ValueError(
            f"model_type '{model_type}' không hợp lệ. "
            f"Chọn từ: {all_types}"
        )


if __name__ == "__main__":
    # Test model creation
    print("Testing Model Creation...")

    # Test Conv1D-GRU
    print("\n1. Testing Conv1D-GRU Model:")
    model1 = create_model('conv1d_gru', input_steps=50, output_steps=5)

    # Test Conv1D
    print("\n2. Testing Conv1D Model:")
    model2 = create_model('conv1d', input_steps=50, output_steps=5)

    # Test GRU
    print("\n3. Testing GRU Model:")
    model3 = create_model('gru', input_steps=50, output_steps=5)

    print("\n✓ All models created successfully!")
