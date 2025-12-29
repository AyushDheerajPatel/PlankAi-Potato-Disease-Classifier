from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import logging
import tflite_runtime.interpreter as tflite

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flashing messages

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create upload folder if it doesn't exist
os.makedirs('static', exist_ok=True)

# Load the TFLite model and allocate tensors
try:
    interpreter = tflite.Interpreter(model_path='potato_model.tflite')
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.allocate_tensors()
    logger.info("TFLite model loaded successfully")
except Exception as e:
    logger.exception("Error loading TFLite model")
    interpreter = None

# Class labels for potato diseases
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Information and treatment guidance for each disease
disease_info = {
    'Potato___Early_blight': {
        'name': 'Early Blight',
        'description': 'Early blight is a common fungal disease that causes dark, concentric lesions on older leaves and tubers. It spreads quickly in warm, wet conditions.',
        'treatment': 'Remove and destroy infected leaves, avoid overhead irrigation, rotate crops, and apply appropriate fungicides (e.g., chlorothalonil or mancozeb) following label instructions.'
    },
    'Potato___Late_blight': {
        'name': 'Late Blight',
        'description': 'Late blight is a serious disease caused by Phytophthora infestans; it causes water-soaked lesions that rapidly turn brown and can destroy plants and tubers.',
        'treatment': 'Immediately remove and destroy infected plants. Use certified disease-free seed, improve air circulation, avoid overhead watering, and apply recommended fungicides (e.g., mancozeb, copper-based products) as advised by local extension services.'
    },
    'Potato___healthy': {
        'name': 'Healthy',
        'description': 'No disease detected. The plant appears healthy.',
        'treatment': 'Maintain good cultural practices: crop rotation, balanced fertilization, adequate watering, and monitor regularly for signs of disease.'
    }
}

def preprocess_image(img_path):
    """Preprocess the image for model prediction"""
    try:
        img = Image.open(img_path)
        img = img.resize((255, 255))
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Error preprocessing image: {str(e)}")
        return None

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('fixed_index.html', message='No file selected')
        
        file = request.files['file']
        if file.filename == '':
            return render_template('fixed_index.html', message='No file selected')

        if file and allowed_file(file.filename):
            try:
                # Save the file
                filename = secure_filename(file.filename)
                file_path = os.path.join('static', filename)
                file.save(file_path)
                
                # Preprocess the image
                processed_image = preprocess_image(file_path)
                if processed_image is None:
                    return render_template('fixed_index.html', message='Error processing image')

                # Make prediction
                if interpreter is None:
                    logger.error("Prediction requested but model is not loaded")
                    return render_template('fixed_index.html', message='Model not loaded. Please ensure the model file is present.')

                # Pad the input to a batch of 32
                input_data = np.tile(processed_image, (32, 1, 1, 1))

                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                predictions = interpreter.get_tensor(output_details[0]['index'])
                
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = round(float(np.max(predictions[0])) * 100, 2)
                
                # Fetch description/treatment for the predicted class
                info = disease_info.get(predicted_class, {})
                disease_name = info.get('name', predicted_class)
                disease_description = info.get('description', 'No information available.')
                disease_treatment = info.get('treatment', 'No treatment information available.')

                return render_template('fixed_index.html',
                                    image_path=file_path,
                                    predicted_label=disease_name,
                                    confidence=confidence,
                                    disease_description=disease_description,
                                    disease_treatment=disease_treatment)

            except Exception as e:
                logger.error(f"Error during prediction: {str(e)}")
                return render_template('fixed_index.html', message='Error during prediction')
        else:
            return render_template('fixed_index.html', message='Invalid file type. Please upload a PNG, JPG, or JPEG image.')

    return render_template('fixed_index.html')

if __name__ == '__main__':
    app.run(debug=True)