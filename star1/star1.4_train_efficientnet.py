import tensorflow as tf
from tensorflow.keras import layers, Model
#from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.applications import EfficientNetB0
import os
import matplotlib.pyplot as plt 


image_size = (224, 224)
batch_size = 32
epoch = 15
layers_unfreeze = 80
initial_learning_rate = 0.0001
learning_rate = 0.0001

#data augmentation step
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 40,
    width_shift_range = 0.3,
    height_shift_range = 0.3,
    horizontal_flip = True,
    zoom_range = 0.2,
    brightness_range = [0.8, 1.2],
    shear_range = 0.2,
    fill_mode = 'nearest'
)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle=True
)
val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle=False
)

num_classes = len(train_data.class_indices)

print("Building on EfficientNet - phase 1")

base_model = EfficientNetB0(
    weights = 'imagenet',
    include_top = False,
    pooling = 'avg',
    input_shape = (224, 224, 3)
)

base_model.trainable = False 

inputs = layers.Input(shape=(224, 224, 3))
x = base_model(inputs, training=False)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer = tf.keras.regularizers.l2(0.001))(x)

#asking the keras tensorflow to form an entire new network here 
model = Model(inputs, outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#training the data
history1 = model.fit (
    train_data,
    epochs = epoch,
    validation_data = val_data,
    verbose = 1
)

model.save('models/efficient_net1_star4.keras')
print("Phase 2 - Aggressive Fine Tuning")

#loading the saved model

model = tf.keras.load_model('models/efficientnet_net1_star4.keras')

initial_loss, initial_acc = model.evaluate(val_data, verbose=0)
base_model2 = model.layers[1]
base_model2.traintable = True
freeze_till = len(base_model2.layers) - layers_unfreeze

#freezing and unfreezing the ones that we truly want 
for i, layer in enumerate(base_model.layers):
    if i < freeze_till:
        layer.trainable = False #notice it is 'layer.trainabale' not 'base_model'.trainbale {we came to a more granular level}
    else:
        layer.trainable = True

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
total_count = model.count_params()

model_input = model.input

x = base_model(model_input, training=False)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

model = Model(model_input, outputs)

lr_schedule = ExponentialDecay(
    initial_learning_rate = initial_learning_rate,
    decay_steps = 1000,
    decay_rate = 0.96
)

early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 5, 
    restore_best_weight = True,
    verbose = 1
)

os.makedirs('models', exist_ok = True)
model_checkpoint = ModelCheckpoint(
    'models/flower_classifier_star3_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 3,
    min_lr = 1e-7,
    verbose = 1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("Compiling all those models")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs = epoch,
    validation_data = val_data,
    callbacks=callbacks,
    verbose=1
)

final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
best_val_acc = max(history.history['val_accuracy']) * 100

print(f"\n  Star 2 baseline: {initial_acc*100:.2f}%")
print(f"  Star 3 final val accuracy: {final_val_acc:.2f}%")
print(f"  Star 3 BEST val accuracy: {best_val_acc:.2f}%")
print(f"  Improvement: +{best_val_acc - initial_acc*100:.2f}%")



model.save('models/star4_train_efficientnetnet.keras')
print(" Saved: models/flower_classifier_star3_final.keras")

# Save class names
with open('models/class_names.txt', 'w') as f:
    for class_name in train_data.class_indices.keys():
        f.write(class_name + '\n')
print("  ✓ Saved: models/class_names.txt")


print("\n Creating training plots...")

fig = plt.figure(figsize=(16, 6))

# Accuracy plot
plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
plt.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
plt.axhline(y=initial_acc, color='g', linestyle='--', alpha=0.7, 
            label=f'Star 2 baseline ({initial_acc*100:.1f}%)')
plt.axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, 
            label='Target (95%)')
plt.title('Star 3: Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], 'b-', label='Training', linewidth=2)
plt.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
plt.title('Star 3: Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# Learning rate plot (if available)
plt.subplot(1, 3, 3)
if 'lr' in history.history:
    plt.plot(history.history['lr'], 'purple', linewidth=2)
    plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
else:
    # Create comparison bar chart
    stages = ['Star 1\n(Feature\nExtraction)', 'Star 2\n(Partial\nFine-Tuning)', 'Star 3\n(Aggressive\nFine-Tuning)']
    accuracies = [88.67, 92.89, best_val_acc]
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    bars = plt.bar(stages, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    plt.axhline(y=95, color='orange', linestyle='--', linewidth=2, label='Target (95%)')
    plt.title('Progression Through Stars', fontsize=14, fontweight='bold')
    plt.ylabel('Validation Accuracy (%)')
    plt.ylim([85, 100])
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.2f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('training_star3.png', dpi=150)
print("  ✓ Saved: training_star3.png")


