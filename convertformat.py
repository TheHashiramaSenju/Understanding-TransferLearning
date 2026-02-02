import tensorflow as tf
print("keras conversion - we load the model and then save in different")

model = tf.keras.models.load_model('models/flower_classifier.h5')
model.save('models/flower_classifier.keras')
print("Keras model has been saved")

print("We are one with both formats")