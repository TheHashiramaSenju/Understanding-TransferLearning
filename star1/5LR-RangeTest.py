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

class base_model_tuning():
    def __init__(self, warmup_epochs, ):