# test_model.py
import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

print("Flower Classifier - Test Script\n")

# Check if model exists
if not os.path.exists('models/flower_classifier.h5'):
    print("Model not found! Train the model first:")
    print("   python train_model.py")
    exit(1)

# Load model
print(" Loading model...")
model = tf.keras.models.load_model('models/flower_classifier.h5')
print("✓ Model loaded!\n")

# Load class names
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

def predict_flower(image_path):
    """Predict flower in image"""
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f" Error loading image: {e}")
        return
    
    # Preprocess
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    predictions = model.predict(img_array, verbose=0)
    
    # Results
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx] * 100

    print(f" Image: {image_path}")
    print(f" PREDICTION: {class_names[predicted_idx].upper()}")
    print(f"   Confidence: {confidence:.1f}%\n")
    
    print("All probabilities:")
    for i, name in enumerate(class_names):
        bar = "█" * int(predictions[0][i] * 50)
        print(f"  {name:12s} {predictions[0][i]*100:5.1f}% {bar}")
    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_model.py <image_path>")
        print("\nExample:")
        print("  python test_model.py flowers/rose/image_001.jpg")
        print("  python test_model.py test_images/my_flower.jpg")
    else:
        predict_flower(sys.argv[1])
