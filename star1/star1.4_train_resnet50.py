import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import os 

image_size = (224, 224)
batch_size = 32
epoch = 15
layers_unfreeze = 60

#data augmentation step
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range = 0.2,
    fill_mode = 'nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

#dataloading and validation
train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
) 

val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_data.class_indices)

print("Resnet building")

base_model = ResNet50(
    weights='imagenet',
    include_top=False,
    pooling='avg',
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training = False)
x = layers.Dense(128, activation='relu',kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

#asking the keras layer to form a complete model combining our custom layers after freezing
 
model = Model(inputs, outputs)
#compiling the model into a complete one now
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss = 'categorical_crossentropy', 
    metrics=['accuracy']    
)

#training the data
history1 = model.fit(
    train_data,
    epochs=epoch,
    validation_data=val_data,
    verbose=1    
)

#let us fine tune this model !

print("this is fine tuning")
base_model.trainable = True

total_layers = len(base_model.layers)
freeze_till = total_layers - layers_unfreeze

for i, layer in enumerate(base_model.layers):
    if i < freeze_till:
        layer.trainable = False
    else:
        layer.trainable = True


print(f"  Unfrozen: {layers_unfreeze} layers")
print(f"  Frozen: {freeze_till} layers")

lr_schedule = ExponentialDecay(0.0001, decay_steps = 1000, decay_rate=0.96)

early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

model_checkpoint = ModelCheckpoint(
    'models/star4_resnet50_best.keras',
    monitor = 'val_accuracy',
    save_best_only = True,
    verbose = 1
)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("\n🔧 Recompiling with scheduled learning rate...")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    epochs=epoch,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

best_val_acc = max(history2.history['val_accuracy']) * 100
print(f"\n ResNet50 best validation accuracy: {best_val_acc:.2f}%")

# Save final model
model.save('models/star4_resnet50_final.keras')
print("💾 Saved: models/star4_resnet50_final.keras")