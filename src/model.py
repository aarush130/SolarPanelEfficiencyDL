"""
Deep Learning Model Architecture
================================
Neural network architectures for solar panel efficiency prediction.
Includes multiple model variants and custom layers.

Author: Solar Panel Efficiency Research Team
Version: 1.0.0
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, 
    TensorBoard, LearningRateScheduler
)
import numpy as np
from typing import Tuple, List, Optional, Dict
import os


class AttentionLayer(layers.Layer):
    """
    Custom Attention Layer for feature importance learning.
    
    This layer learns to weight input features based on their importance
    for the prediction task.
    """
    
    def __init__(self, units: int = 32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.units = units
        
    def build(self, input_shape):
        self.W = self.add_weight(
            name='attention_weight',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        self.b = self.add_weight(
            name='attention_bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        self.u = self.add_weight(
            name='attention_context',
            shape=(self.units, 1),
            initializer='glorot_uniform',
            trainable=True
        )
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # Compute attention scores
        score = tf.nn.tanh(tf.matmul(x, self.W) + self.b)
        attention_weights = tf.nn.softmax(tf.matmul(score, self.u), axis=1)
        
        # Apply attention weights
        context_vector = x * attention_weights
        return context_vector
    
    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update({'units': self.units})
        return config


class ResidualBlock(layers.Layer):
    """
    Residual Block with skip connections for deep networks.
    """
    
    def __init__(self, units: int, dropout_rate: float = 0.2, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.dense1 = layers.Dense(
            self.units, 
            activation='relu',
            kernel_regularizer=regularizers.l2(0.001)
        )
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(self.dropout_rate)
        
        self.dense2 = layers.Dense(
            self.units,
            kernel_regularizer=regularizers.l2(0.001)
        )
        self.bn2 = layers.BatchNormalization()
        
        # Projection layer if dimensions don't match
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units)
        else:
            self.projection = None
            
        super(ResidualBlock, self).build(input_shape)
        
    def call(self, inputs, training=None):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        
        # Skip connection
        if self.projection is not None:
            inputs = self.projection(inputs)
        
        return tf.nn.relu(x + inputs)
    
    def get_config(self):
        config = super(ResidualBlock, self).get_config()
        config.update({
            'units': self.units,
            'dropout_rate': self.dropout_rate
        })
        return config


class SolarPanelEfficiencyModel:
    """
    Deep Learning Model Factory for Solar Panel Efficiency Prediction.
    
    Supports multiple architectures:
    - 'standard': Standard feedforward neural network
    - 'deep': Deep network with residual connections
    - 'attention': Network with attention mechanism
    - 'ensemble': Combination of multiple architectures
    """
    
    def __init__(self, input_dim: int, model_type: str = 'deep'):
        """
        Initialize the model factory.
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        model_type : str
            Type of model architecture
        """
        self.input_dim = input_dim
        self.model_type = model_type
        self.model = None
        self.history = None
        
    def build_standard_model(self, 
                            hidden_layers: List[int] = [256, 128, 64, 32],
                            dropout_rate: float = 0.3,
                            l2_reg: float = 0.001) -> Model:
        """
        Build a standard feedforward neural network.
        """
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')
        
        x = inputs
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(l2_reg),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='StandardModel')
        return model
    
    def build_deep_residual_model(self,
                                  residual_blocks: int = 4,
                                  units_per_block: int = 128,
                                  dropout_rate: float = 0.2) -> Model:
        """
        Build a deep network with residual connections.
        """
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')
        
        # Initial dense layer to match residual block dimensions
        x = layers.Dense(units_per_block, activation='relu', name='initial_dense')(inputs)
        x = layers.BatchNormalization(name='initial_bn')(x)
        
        # Residual blocks
        for i in range(residual_blocks):
            x = ResidualBlock(units_per_block, dropout_rate, name=f'residual_block_{i+1}')(x)
        
        # Output layers
        x = layers.Dense(64, activation='relu', name='pre_output')(x)
        x = layers.Dropout(0.2, name='output_dropout')(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='DeepResidualModel')
        return model
    
    def build_attention_model(self,
                             attention_units: int = 64,
                             hidden_layers: List[int] = [128, 64],
                             dropout_rate: float = 0.3) -> Model:
        """
        Build a network with attention mechanism.
        """
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')
        
        # Reshape for attention
        x = layers.Reshape((self.input_dim, 1))(inputs)
        x = layers.Dense(32, activation='relu')(x)
        
        # Apply attention
        x = AttentionLayer(attention_units, name='attention')(x)
        x = layers.Flatten()(x)
        
        # Hidden layers
        for i, units in enumerate(hidden_layers):
            x = layers.Dense(
                units, 
                activation='relu',
                kernel_regularizer=regularizers.l2(0.001),
                name=f'dense_{i+1}'
            )(x)
            x = layers.BatchNormalization(name=f'bn_{i+1}')(x)
            x = layers.Dropout(dropout_rate, name=f'dropout_{i+1}')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionModel')
        return model
    
    def build_ensemble_model(self) -> Model:
        """
        Build an ensemble model combining multiple architectures.
        """
        inputs = layers.Input(shape=(self.input_dim,), name='input_layer')
        
        # Branch 1: Standard feedforward
        branch1 = layers.Dense(128, activation='relu')(inputs)
        branch1 = layers.BatchNormalization()(branch1)
        branch1 = layers.Dropout(0.3)(branch1)
        branch1 = layers.Dense(64, activation='relu')(branch1)
        branch1 = layers.Dense(32, activation='relu')(branch1)
        
        # Branch 2: Deeper network
        branch2 = layers.Dense(256, activation='relu')(inputs)
        branch2 = layers.BatchNormalization()(branch2)
        branch2 = layers.Dropout(0.2)(branch2)
        branch2 = layers.Dense(128, activation='relu')(branch2)
        branch2 = layers.Dense(64, activation='relu')(branch2)
        branch2 = layers.Dense(32, activation='relu')(branch2)
        
        # Branch 3: Wide network
        branch3 = layers.Dense(512, activation='relu')(inputs)
        branch3 = layers.BatchNormalization()(branch3)
        branch3 = layers.Dropout(0.4)(branch3)
        branch3 = layers.Dense(32, activation='relu')(branch3)
        
        # Concatenate branches
        merged = layers.Concatenate()([branch1, branch2, branch3])
        
        # Final layers
        x = layers.Dense(64, activation='relu')(merged)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='EnsembleModel')
        return model
    
    def build(self, **kwargs) -> Model:
        """
        Build the model based on the specified type.
        """
        builders = {
            'standard': self.build_standard_model,
            'deep': self.build_deep_residual_model,
            'attention': self.build_attention_model,
            'ensemble': self.build_ensemble_model
        }
        
        if self.model_type not in builders:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Available types: {list(builders.keys())}")
        
        self.model = builders[self.model_type](**kwargs)
        print(f"Built {self.model_type} model with {self.model.count_params():,} parameters")
        return self.model
    
    def compile(self, 
                learning_rate: float = 0.001,
                loss: str = 'mse',
                metrics: List[str] = ['mae', 'mse']) -> None:
        """
        Compile the model with optimizer and loss function.
        """
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        print(f"Model compiled with learning rate: {learning_rate}")
    
    def get_callbacks(self, 
                     model_path: str = 'models/best_model.keras',
                     patience: int = 15,
                     log_dir: str = 'logs') -> List:
        """
        Get training callbacks.
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            TensorBoard(
                log_dir=log_dir,
                histogram_freq=1
            )
        ]
        return callbacks
    
    def summary(self) -> None:
        """Print model summary."""
        if self.model is not None:
            self.model.summary()
        else:
            print("Model not built yet. Call build() first.")


def create_model(input_dim: int, 
                model_type: str = 'deep',
                **kwargs) -> Tuple[Model, SolarPanelEfficiencyModel]:
    """
    Factory function to create and compile a model.
    
    Parameters:
    -----------
    input_dim : int
        Number of input features
    model_type : str
        Type of model to create
    **kwargs
        Additional arguments passed to the model builder
        
    Returns:
    --------
    tuple
        (compiled_model, model_factory)
    """
    factory = SolarPanelEfficiencyModel(input_dim, model_type)
    model = factory.build(**kwargs)
    factory.compile()
    
    return model, factory


if __name__ == "__main__":
    # Test model creation
    input_dim = 17  # Number of features after engineering
    
    print("=" * 60)
    print("Testing Model Architectures")
    print("=" * 60)
    
    for model_type in ['standard', 'deep', 'attention', 'ensemble']:
        print(f"\n{model_type.upper()} MODEL:")
        print("-" * 40)
        model, factory = create_model(input_dim, model_type)
        factory.summary()
