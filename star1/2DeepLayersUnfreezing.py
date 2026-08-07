#here we will tweak even more parameters 
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os


image_size = (224, 224)
batch_size = 32
epochs = 10
learning_rate = 0.0001
layers_unfreeze = 30

print("We are starting fine tuning here")


val_datagen = ImageDataGenerator(rescale=1./255)
train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range =0.2, 
    horizontal_flip = True,
    fill_mode = 'nearest'
)


train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = image_size,
    batch_size = batch_size,
    class_mode='categorical',
    #fill_mode = 'nearest',
    shuffle = True
)

val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle=False
) 

#model-loading 
print("Loading the pre-existent modules now")
model = tf.keras.models.load_model('models/flower_classifier.h5')

initial_loss, initial_acc = model.evaluate(val_data, verbose=0)
print(f"The starting accuracy is {initial_acc*100:.2f}% from 1 star")


if initial_acc < 0.80:
    print("WARNING: 1-star accuracy seems low! Check if correct model loaded")
    print(f"Expected: ~88%, Got: {initial_acc*100:.1f}%")

#unfreezing the top layers 
print(f"Unfreezing {layers_unfreeze}")
loaded_model = model
base_model = loaded_model.layers[1]

print(f"  Base model: {base_model.name}")
print(f"  Total layers in base: {len(base_model.layers)}")

#Let us UNFREEZE the bease models
base_model.trainable = True

#UNFREEZING EVERYTHING might kill the point. so we try to unfreeze only till a specific layer here 

freeze_till = len(base_model.layers) - layers_unfreeze

for i, layer in enumerate(base_model.layers):
    if i < freeze_till:
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum([tf.size(w).numpy() for w in model.trainable_weights])
total_count = model.count_params()

print("Model Parameters")
print(f"Total {total_count:,}")
print(f"  Trainable: {trainable_count:,} ({trainable_count/total_count*100:.1f}%)")
print(f"  Frozen: {total_count - trainable_count:,}") 

print("Compiling")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    loss = 'categorical_crossentropy',
    metrics = ['accuracy']
)

#finetuning(fitting our compiled model)
history = model.fit(
    train_data,
    epochs=epochs,
    validation_data=val_data,
    verbose=1
)

final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100

if final_val_acc >= 92:
    print("\n EXCELLENT! 2-star level achieved!")
elif final_val_acc >= 90:
    print("\n GREAT! Almost there!")
else:
    print("\n Expected 92%+, consider training more epochs")


os.makedirs('models', exist_ok=True)
model.save('models/flower_classifier_finetuned.keras')

with open('models/class_names.txt', 'w') as f:
    for class_name in train_data.class_indices.keys():
        f.write(class_name + '\n')

fig = plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Training', linewidth=2)
plt.plot(history.history['val_accuracy'], 'r-', label='Validation', linewidth=2)
plt.axhline(y=initial_acc, color='g', linestyle='--', alpha=0.7, 
            label=f'1-star baseline ({initial_acc*100:.1f}%)')
plt.title('Fine-Tuning Accuracy (2-Star)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-', label='Training', linewidth=2)
plt.plot(history.history['val_loss'], 'r-', label='Validation', linewidth=2)
plt.title('Fine-Tuning Loss (2-Star)', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_2star.png', dpi=150)
print(f" Training plot: training_2star.png")
