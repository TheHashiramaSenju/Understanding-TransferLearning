import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import os

#dataset existence 
if not os.path.exists('dataset/train'):
    print("The dataset path seems to be incorrect./ non existent")

image_size = (224, 224)
batch_size = 32
epochs = 20

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range=0.2,
    horizontal_flip = True,
    fill_mode = 'nearest'    
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size = image_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    shuffle=True
)

val_data = val_datagen.flow_from_directory(
    'dataset/validation',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

#important point to note : Here we are twisting and training the train data. to reduce any overfitting/underfitting challenges that might arise
#but here we dont do much of twisting of validation data because of the testing data nature it posses


print("Building Transfer Learning")
base_model = MobileNetV2(
    weights='imagenet',
    include_top=False, #freezing the layers and deleting the last layer 
    pooling='avg',
    input_shape=(224, 224, 3)
)

base_model.trainable = False #layer freezing 

#adding input to the neural layer 
inputs = layers.Input(shape=(224, 224, 3))
#unput given to the base model 
#that initially set frozen model, now makes it differently from this freezen model in such a way that it updates all the batch normalisation layers
x = base_model(inputs, training=False)
#now the base model is trained and ready to give the outputs to the next layers, in which we are gonnna add now
x = layers.Dense(128, activation='relu')(x) #(x) means the output from the base model and so on the next x means the next outputs from the other models
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation='softmax')(x)

model = Model(inputs, outputs)

model.compile(
    optimizer = tf.keras.optimizers.Adam(0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  Total parameters: {model.count_params():,}")
trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])

#training the model
print("WE are TRAINING the BUILT model (CUSTOM MODEL)")
history = model.fit(train_data, epochs=epochs, validation_data=val_data, verbose=1)


final_train_acc = history.history['accuracy'][-1] * 100
final_val_acc = history.history['val_accuracy'][-1] * 100 #find what this actually does here !

if final_val_acc > 85:
    model.save('flower_classifier_model.h5')
    print(f"Model trained and saved with final training accuracy: {final_train_acc:.2f}% and validation accuracy: {final_val_acc:.2f}%")
    print("Model saved as 'flower_classifier_model.h5'")

elif final_val_acc < 75:
    print(f"Model training needs improvement. Final training accuracy: {final_train_acc:.2f}%, validation accuracy: {final_val_acc:.2f}%")
    print("Consider tuning hyperparameters or using a different architecture.")

else:
    print(f"Model trained with final training accuracy: {final_train_acc:.2f}% and validation accuracy: {final_val_acc:.2f}%")
    print("Consider further training or fine-tuning for better performance.")
    
#model saving 
os.makedirs('models', exist_ok=True)
model.save('models/flower_classifier.h5')
#saving class names 
with open('models/class_names.txt', 'w') as f:
    for class_name in train_data.class_indices.keys():
        f.write(class_name +'\n')
print(f"Class names saved at models/class_names.txt ")

#plotting the reusults 
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], 'b-', label='Training')
plt.plot(history.history['val_accuracy'], 'r-', label='Validation')
plt.title('Model Accuracy', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

#loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], 'b-', label='Training')
plt.plot(history.history['val_loss'], 'r-', label='Validation')
plt.title('Model Loss', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('training_history2.png', dpi=150)
print(f"Training plot saved: training_history2.png")

