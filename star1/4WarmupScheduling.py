#here we will watch how we warmup the learning rates of the model that we want to train !
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocesssing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import os

image_size = (224, 224)
batch_size = 32
epochs = 20
learning_rate = 0.0001
layers_unfreeze = 80
initial_learning_rate = 0.0001


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
    class_mode = 'categorical', #explore why !
    shuffle=True
)
val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = True 
)


def base_scratch_model():
    print("Importing the base model for analysis")
    model = MobileNetV2(
        weights='imagenet',
        include_top = False, 
        pooling = 'avg',
        input_shape = (224, 224, 3) #explore more parameters here
    )
    #setting the base_model as true here we will get something like 
    model.trainable = True
        
    print(f"Unfreezing the top layers inside the models")
    base_model = model
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
    
    #understand the encapsulation of the trained classes
    
    trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
    total_count = model.count_params()
    
    print()
    print("Take a closer look here")
    print()
    
    print("Model Parameters")
    print(f"Total {total_count :,}")
    print(f"  Trainable: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)") #explore more about this part here and make sure to understand everything here 
    print(f" Frozen: {total_count - trainable_count}")   
    
    print()
    print("Pay attention here")
    print()
    
    print(f"  Base model: {base_model.name}")
    print(f"  Total layers: {len(base_model.layers)}")
    print(f"  Frozen: Layers 0-{freeze_till-1} ({freeze_till} layers)")
    print(f"  Unfrozen: Layers {freeze_till}-{len(base_model.layers)-1} ({layers_unfreeze} layers)")
    print(f"\n  Parameters:")
    print(f"    Total: {total_count:,}")
    print(f"    Trainable: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
    print(f"    Frozen: {total_count - trainable_count:,} ({(total_count-trainable_count)/total_count*100:.1f}%)")
    
    print()
    print("Understand the metrics and have your head wrapped around this concept")
    
    class WarmupScheduling(tf.keras.callbacks.Callback):
        def __init__(self, warmup_poch, target_lr, initial_lr=None ):
            super().__init__()
            warmup_epoch = warmup_epoch
            self.target_lr = target_lr
            self.initial_lr = initial_lr if initial_lr else target_lr/100
            self.lr_history = [] #OOPs concept - self. and return values !
            
        def on_train_begin(self, logs = None):
            tf.keras.backend.set_value(
                self.model.optimizer.lr, self.initial_lr   
            )
        
        def on_epoch_begin(self, epoch, logs = None):
            
            if epoch < self.target_lr:
                progress = (epoch+1)/self.warmup_epoch
                lr = self.initial_lr + (self.target_lr - self.initial_lr) * progress  #alternative Linear Formula + what if outside and how to program it in the outside without essentially changing the values here
            else:
                lr = self.target_lr
            
            #now here is where the LR actually gets set here
            tf.keras.backend.set_value(self.model.optimizer.lr, lr)
            self.lr_history.append(lr)
       
        def on_epoch_end(self, epoch, logs = None):
            logs = logs or {}
            logs['lr'] = tf.keras.backend.get_value(self.model.optimizer.lr)

    
    lr_scheduler = ExponentialDecay( #about this, explore the other various options and get the  fitting options
        initial_learning_rate=initial_learning_rate,
        decay_steps = 1000, #understand the nature of this variable 
        decay_rate = 0.96
    )
    
    #callbacks to save Resources 
    early_stopping = EarlyStopping(
        monitor = 'val_accuracy', #check other variables here too 
        patience = 5, 
        restore_best_weights = True
    )
    
    os.makedirs('models', exist_ok=True)
    model_checkpoint = ModelCheckpoint(
        'models/flower_classifier_star3_best.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
    print("Model has done checkpointing")
    
    reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.5, 
        patience = 3,
        min_lr = 1e-7,
    )
    callbacks = [WarmupScheduling,lr_scheduler, early_stopping, model_checkpoint, reduce_lr]
    
    history = model.fit(
        train_data,
        epochs = epochs,
        validation_data = val_data,
        callbacks = callbacks,
    )
    