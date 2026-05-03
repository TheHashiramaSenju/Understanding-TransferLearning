#here we will watch how we warmup the learning rates of the model that we want to train !
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocesssing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os
import numpy as np
from tkinter import filedialog, Tk

# Interactive file opener for model selection
root = Tk()
root.withdraw()  # Hide the root window
model_path = filedialog.askopenfilename(
    title="Select a Keras model file",
    filetypes=[("Keras Model Files", "*.keras *.h5"), ("All Files", "*.*")]
)
root.destroy()

if not model_path:
    raise ValueError("No model file selected. Exiting.")

FINETUNER = tf.keras.models.load_model(model_path)
image_size = (224, 224)
batch_size = 32
epochs = 20
learning_rate = 0.0001
initial_learning_rate = 0.0001
fine_tuner = FINETUNER


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

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical' #explore why !
    shuffle=True
)
val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True 
)

model = MobileNetV2(
        weights = 'imagenet',
        include_top = False,
        pooling = 'avg',
        input_shape = (224, 224, 3)
)

def plug_in_model(model):
    
    model.trainable = True
    inputs = layers.Input(shape = (224, 224, 3))
    x = model(inputs, training = True)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(5, activation='softmax')(x)
    model = Model(inputs, outputs)
    
def unfreezeer():
    
    def basic_unfreezing(plug_in_model, layers_unfreeze = 2): 
           
        print(f"Unfreezing the top layers inside the models")
        base_model = plug_in_model
        freeze_till = len(base_model.layers) - layers_unfreeze

        for i, layer in enumerate(base_model.layers):
            if i < freeze_till:
                layer.trainable = False
            if i > freeze_till:
                layer.trainable = True

        inputs = layers.Input(shape=(224,224,3))
        x = model(inputs, training=True) #search more about this
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(5, activation='softmax')(x)


        model = Model(inputs, outputs)
        trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
        total_count = model.count_params()
    
    def fine_tune_unfreezing(fine_tuner, freeze_till):
        
        print(f"You are using the fine tuning script and the chosen model is {fine_tuner}")
        base_model = fine_tuner[1]
        base_model.trainable = True
        freeze_till = len(base_model.layers) - freeze_till
        
        #unfreezing more layers inside the model
        
        for i, layer in enumerate(base_model.layers):
            if i < freeze_till:
                layer.Trainable = False
            else:
                layer.Trainable  = True 
                
        num_classes = len(train_data.class_indices)
        fine_tuner_input = fine_tuner.input
        x = base_model(fine_tuner_input, training = False)
        x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        model = Model(fine_tuner_input, outputs)
        
def schedulersandstuffs():
     
    def lr_scheduler(required_lr, decay_steps, decay_rate):
        
        lr_schedule = ExponentialDecay(
            initial_learning_rate=required_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate
        )

    
    class WarmupCosine(tf.keras.callbacks.Callback):
        def __init__(self, warmup_epochs, eta_max, eta_min = 1e-7, T0 = 10, TMul = 2):
            super().__init__()
            self.warmup_epochs = warmup_epochs
            self.eta_max = eta_max
            self.eta_min = eta_min
            self.T0 = T0
            self.TMul = TMul
            self.lr_history = []
            
            progress = (epochs + 1 ) / self.warmup_epochs
            def cosine_lr(self, progress):
                return self.eta_min + 0.5 * (self.eta_max - self.eta_min) *(1 + np.cos(np.pi * progress))
            
            def on_train_begin(self, epoch, logs = None):
                tf.keras.backend.set_value(
                    self.model.optimizer.lr, self.initial_lr
                )
                
            def on_epoch_begin(self, epoch, logs = None):
                if epoch < self.target_lr:
                    lr = cosine_lr(progress)
                else:
                    lr  = 1e-2
                
                tf.keras.backend.set_value(self.model.optimizer.lr, lr)
                self.lr_history.append(lr)
                
            def on_epoch_end(self, epochs, logs = None):
                logs = logs or {}

def callbackmodules():
    
    def early_stopping(monitor, patience, restore_best_weights = True, verbose = 1 ):
        return EarlyStopping(
            monitor = 'val_accuracy',
            patience = patience,
            restore_best_weights = restore_best_weights,
            verbose = verbose
        )
    def model_checkpointing(monitor='val_accuracy', save_best_only=True, verbose=1):
        return ModelCheckpoint(
            'models/flower_classifier_star3_best.keras',
            monitor=monitor,
            save_best_only=save_best_only,
            verbose=verbose
        )
    def reduce_lr_on_plateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1):
        return ReduceLROnPlateau(
            monitor=monitor,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=verbose
        )
        


