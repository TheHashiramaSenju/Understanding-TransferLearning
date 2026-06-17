#! Modular, Plug-and-Play Transfer Learning Training Script
"""
This module provides a modular, reusable framework for transfer learning with fine-tuning.
Components can be easily swapped, configured, and combined for different training scenarios.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os
import numpy as np
from tkinter import filedialog, Tk
from typing import Tuple, Optional, Dict


class DataHandler:
    """Handles data loading and preprocessing."""
    
    def __init__(self, image_size: Tuple[int, int] = (224, 224), batch_size: int = 32):
        self.image_size = image_size
        self.batch_size = batch_size
    
    def create_data_generators(self) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
        """Create training and validation data generators."""
        val_datagen = ImageDataGenerator(rescale=1./255)
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.3,
            height_shift_range=0.3,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2],
            shear_range=0.2,
            fill_mode='nearest'
        )
        return train_datagen, val_datagen
    
    def load_data(self, train_path: str, val_path: str):
        """Load training and validation data."""
        train_datagen, val_datagen = self.create_data_generators()
        
        train_data = train_datagen.flow_from_directory(
            train_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical',
            shuffle=True
        )
        
        val_data = val_datagen.flow_from_directory(
            val_path,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode="categorical",
            shuffle=True
        )
        
        return train_data, val_data



class ModelBuilder:
    """Builds and configures transfer learning models."""
    
    @staticmethod
    def create_base_mobilenetv2(input_shape: Tuple[int, int, int] = (224, 224, 3)):
        """Create base MobileNetV2 model."""
        return MobileNetV2(
            weights='imagenet',
            include_top=False,
            pooling='avg',
            input_shape=input_shape
        )
    
    @staticmethod
    def add_custom_head(base_model, num_classes: int = 5, dropout_rate: float = 0.3) -> Model:
        """Add custom dense layers on top of base model."""
        inputs = layers.Input(shape=(224, 224, 3))
        x = base_model(inputs, training=True)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(dropout_rate)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs)
        return model

class UnfreezeStrategy:
    """Different strategies for unfreezing model layers."""
    
    @staticmethod
    def basic_unfreezing(model: Model, layers_to_unfreeze: int = 2) -> Model:
        """Unfreeze only top N layers."""
        print(f"Unfreezing the top {layers_to_unfreeze} layers in the model")
        base_model = model.layers[0]  # Assuming first layer is the base model
        freeze_till = len(base_model.layers) - layers_to_unfreeze
        
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= freeze_till
        
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_count = model.count_params()
        print(f"Trainable params: {trainable_count} / Total params: {total_count}")
        
        return model
    
    @staticmethod
    def fine_tune_unfreezing(model: Model, freeze_till: int, num_classes: int = 5) -> Model:
        """Fine-tune unfreezing with more aggressive layer unfreezing."""
        print(f"Fine-tuning with freeze_till={freeze_till}")
        base_model = model.layers[0]
        base_model.trainable = True
        freeze_till = len(base_model.layers) - freeze_till
        
        for i, layer in enumerate(base_model.layers):
            layer.trainable = i >= freeze_till
        
        return model

class LearningRateScheduler:
    """Learning rate scheduling strategies."""
    
    @staticmethod
    def exponential_decay(initial_lr: float, decay_steps: int, decay_rate: float):
        """Create exponential decay learning rate schedule."""
        return ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

class WarmupScheduler(tf.keras.callbacks.Callback):
    def __init__(self, warmup_epochs: int, initial_learning):
        ass 
class WarmupCosine(tf.keras.callbacks.Callback):
    """Custom callback for warmup + cosine annealing learning rate schedule."""
    
    def __init__(self, warmup_epochs: int, eta_max: float, eta_min: float = 1e-7, 
                 T0: int = 10, TMul: int = 2):
        super().__init__()
        self.warmup_epochs = warmup_epochs
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.T0 = T0
        self.TMul = TMul
        self.lr_history = []
        self.initial_lr = eta_max
    
    def cosine_lr(self, progress):
        """Calculate cosine annealed learning rate."""
        return self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + np.cos(np.pi * progress))
    
    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.initial_lr)
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            progress = epoch / self.warmup_epochs
            lr = self.cosine_lr(progress)
        else:
            lr = 1e-2
        
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
        self.lr_history.append(lr)



class CallbackFactory:
    """Factory for creating training callbacks."""
    
    @staticmethod
    def early_stopping(monitor: str = 'val_accuracy', patience: int = 5, 
                      restore_best_weights: bool = True, verbose: int = 1):
        return EarlyStopping(
            monitor=monitor,
            patience=patience,
            restore_best_weights=restore_best_weights,
            verbose=verbose
        )
    
    @staticmethod
    def model_checkpoint(filepath: str, monitor: str = 'val_accuracy', 
                        save_best_only: bool = True, verbose: int = 1):
        return ModelCheckpoint(
            filepath,
            monitor=monitor,
            save_best_only=save_best_only,
            verbose=verbose
        )
    
    @staticmethod
    def reduce_lr_on_plateau(monitor: str = 'val_loss', factor: float = 0.5, 
                            patience: int = 3, min_lr: float = 1e-7, verbose: int = 1):
        return ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose
        )
    
    @staticmethod
    def warmup_cosine(warmup_epochs: int, eta_max: float, eta_min: float = 1e-7):
        return WarmupCosine(warmup_epochs, eta_max, eta_min)


class ModelLoader:
    """Interactive model loading from filesystem."""
    
    @staticmethod
    def load_model_interactive() -> str:
        """Open file dialog to select a model."""
        root = Tk()
        root.withdraw()
        model_path = filedialog.askopenfilename(
            title="Select a Keras model file",
            filetypes=[("Keras Model Files", "*.keras *.h5"), ("All Files", "*.*")]
        )
        root.destroy()
        
        if not model_path:
            raise ValueError("No model file selected. Exiting.")
        
        return model_path
    
    @staticmethod
    def load_model(model_path: str):
        """Load model from path."""
        return tf.keras.models.load_model(model_path)

class TrainingOrchestrator:
    """Main orchestrator for training workflow."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize with configuration."""
        self.config = config or self._default_config()
        self.data_handler = DataHandler(
            image_size=tuple(self.config['image_size']),
            batch_size=self.config['batch_size']
        )
        self.model = None
        self.train_data = None
        self.val_data = None
    
    @staticmethod
    def _default_config() -> Dict:
        """Default training configuration."""
        return {
            'image_size': [224, 224],
            'batch_size': 32,
            'epochs': 20,
            'learning_rate': 0.0001,
            'train_path': 'dataset/train',
            'val_path': 'dataset/validation',
            'checkpoint_path': 'models/flower_classifier_star3_best.keras',
        }
    
    def load_data(self):
        """Load training and validation data."""
        print("Loading data...")
        self.train_data, self.val_data = self.data_handler.load_data(
            self.config['train_path'],
            self.config['val_path']
        )
        print(f"Data loaded. Found {len(self.train_data.class_indices)} classes.")
    
    def load_pretrained_model(self):
        """Load a pretrained model interactively."""
        print("Opening file browser...")
        model_path = ModelLoader.load_model_interactive()
        self.model = ModelLoader.load_model(model_path)
        print(f"Model loaded from: {model_path}")
    
    def build_model(self):
        """Build a new transfer learning model."""
        print("Building model...")
        base_model = ModelBuilder.create_base_mobilenetv2()
        num_classes = len(self.train_data.class_indices)
        self.model = ModelBuilder.add_custom_head(base_model, num_classes)
        print(f"Model built with {num_classes} output classes.")
    
    def compile_model(self, optimizer_lr: float = None):
        """Compile the model."""
        lr = optimizer_lr or self.config['learning_rate']
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(f"Model compiled with learning rate: {lr}")
    
    def train(self, callbacks: Optional[list] = None):
        """Train the model."""
        print("Starting training...")
        history = self.model.fit(
            self.train_data,
            validation_data=self.val_data,
            epochs=self.config['epochs'],
            callbacks=callbacks or [],
            verbose=1
        )
        return history
    
    def save_model(self, filepath: str):
        """Save trained model."""
        self.model.save(filepath)
        print(f"Model saved to: {filepath}")
    
    def plot_training_history(self, history):
        """Plot training history."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        axes[0].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].set_title('Model Accuracy')
        
        axes[1].plot(history.history['loss'], label='Train Loss')
        axes[1].plot(history.history['val_loss'], label='Val Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].set_title('Model Loss')
        
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    # Example 1: Train from scratch
    print("=== Example 1: Training from scratch ===\n")
    
    orchestrator = TrainingOrchestrator()
    orchestrator.load_data()
    orchestrator.build_model()
    orchestrator.compile_model()
    
    callbacks = [
        CallbackFactory.early_stopping(patience=5),
        CallbackFactory.model_checkpoint(orchestrator.config['checkpoint_path']),
        CallbackFactory.reduce_lr_on_plateau(),
    ]
    
    # Uncomment to train (commented to avoid long runtime in example)
    # history = orchestrator.train(callbacks)
    # orchestrator.save_model('models/flower_classifier_final.keras')
    # orchestrator.plot_training_history(history)
    
    print("\n✓ Training pipeline ready! Uncomment train() call to start training.\n")
    
    # Example 2: Load pretrained and fine-tune
    print("=== Example 2: Fine-tuning pretrained model ===\n")
    
    # Uncomment to use:
    # orchestrator2 = TrainingOrchestrator()
    # orchestrator2.load_data()
    # orchestrator2.load_pretrained_model()
    # UnfreezeStrategy.fine_tune_unfreezing(orchestrator2.model, freeze_till=10)
    # orchestrator2.compile_model(optimizer_lr=0.00001)
    # history2 = orchestrator2.train(callbacks)
    # orchestrator2.save_model('models/flower_classifier_finetuned_final.keras')
