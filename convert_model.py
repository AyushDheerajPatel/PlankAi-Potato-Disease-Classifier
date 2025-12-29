import tensorflow as tf

# Load the trained Keras model
try:
    model = tf.keras.models.load_model('potato_model.h5', compile=False)
    print("Keras model loaded successfully.")

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    print("Model converted to TFLite format.")

    # Save the TFLite model to a file
    with open('potato_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as potato_model.tflite.")

except Exception as e:
    print(f"An error occurred: {e}")

