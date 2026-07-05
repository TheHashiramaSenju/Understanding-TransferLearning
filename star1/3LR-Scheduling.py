import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau  # NEW
from tensorflow.keras.optimizers.schedules import ExponentialDecay                         # NEW
import matplotlib.pyplot as plt
import os


image_size = (224, 224)
batch_size = 32
epochs = 10
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
    batch_size=batch_size,
    class_mode='categorical',
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
#model loading of 2 star model
model = tf.keras.models.load_model("models/flower_classifier_finetuned.keras")
#evaluation

initial_loss, initial_acc = model.evaluate(val_data, verbose=0)

#this here we use this as the actual fine tuning script

#Unfreezing the layers
base_model = model.layers[1]
base_model.trainable = True
freeze_till = len(base_model.layers) - layers_unfreeze

#freezing the needed ones 
for i, layer in enumerate(base_model.layers):
    if i < freeze_till:
        layer.trainable = False
    else:
        layer.trainable = True
    
#know about connection part. Get checked about the total unfreeze and freeze, get to known abitu the manual adjustment part in here, all possible if we grasp the knowledge of the connection mechanism and the extraciton methods here, learn about how each parameters affects the model 

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
total_count = model.count_params()

#metrics
print(f"  Base model: {base_model.name}")
print(f"  Total layers: {len(base_model.layers)}")
print(f"  Frozen: Layers 0-{freeze_till-1} ({freeze_till} layers)")
print(f"  Unfrozen: Layers {freeze_till}-{len(base_model.layers)-1} ({layers_unfreeze} layers)")
print(f"\n  Parameters:")
print(f"    Total: {total_count:,}")
print(f"    Trainable: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
print(f"    Frozen: {total_count - trainable_count:,} ({(total_count-trainable_count)/total_count*100:.1f}%)")

#here we lern about a new thing again, regularisations!

#input extraction
model_input = model.input

x = base_model(model_input, training=False)
x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

model = Model(model_input, outputs)

#star 3 speciality - lr_schedule = ExponentialDecay

lr_schedule = ExponentialDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=1000,
    decay_rate=0.96  # Multiply LR by 0.96 every 1000 steps
)

#callbacks
early_stopping = EarlyStopping(
    monitor = 'val_accuracy',
    patience = 5,
    restore_best_weights = True,
    verbose = 1
)

os.makedirs('models', exist_ok=True)
model_checkpoint = ModelCheckpoint(
    'models/flower_classifier_star3_best.keras',
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

print("✓ ModelCheckpoint: Save best val_accuracy")

#tips to reduce LR if validation stops working 
reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 3,
    min_lr = 1e-7,
    verbose=1
)

callbacks = [early_stopping, model_checkpoint, reduce_lr]

print("Compiling the model")
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

#model training
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)

# Get final accuracies
final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100
best_val_acc = max(history.history['val_accuracy']) * 100

print(f"\n  Star 2 baseline: {initial_acc*100:.2f}%")
print(f"  Star 3 final val accuracy: {final_val_acc:.2f}%")
print(f"  Star 3 BEST val accuracy: {best_val_acc:.2f}%")
print(f"  Improvement: +{best_val_acc - initial_acc*100:.2f}%")



model.save('models/flower_classifier_star3_final.keras')
print(" Saved: models/flower_classifier_star3_final.keras")

# Save class names
with open('models/class_names.txt', 'w') as f:
    for class_name in train_data.class_indices.keys():
        f.write(class_name + '\n')
print(" Saved: models/class_names.txt")


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
print("Saved: training_star3.png")

