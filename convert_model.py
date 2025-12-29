import tensorflow as tf

# Load the trained Keras model
try:
    model = tf.keras.models.load_model('potato_model.h5', compile=False)
    print("Keras model loaded successfully.")

    # Get the concrete function from the model
    # Specify the input signature to allow for a dynamic batch size
    @tf.function(input_signature=[tf.TensorSpec(shape=[None, 255, 255, 3], dtype=tf.float32)])
    def model_fn(inp):
        return model(inp)

    concrete_fn = model_fn.get_concrete_function()

    # Convert the model to TensorFlow Lite format using the concrete function
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn])
    tflite_model = converter.convert()
    print("Model converted to TFLite format with dynamic batch size.")

    # Save the TFLite model to a file
    with open('potato_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("TFLite model saved as potato_model.tflite.")

except Exception as e:
    print(f"An error occurred: {e}")
