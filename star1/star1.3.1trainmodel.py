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
epochs = 10
learning_rate = 0.0001
layers_unfreeze = 80
initial_learning_rate = 0.0001




