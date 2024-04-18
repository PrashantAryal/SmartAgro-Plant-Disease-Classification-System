from flask import Flask, render_template, request, jsonify
import os
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

app = Flask(__name__)

IMG_WIDTH = 224
IMG_HEIGHT = 224

model_path = "best_of_modelResv2nice.h5"
loaded_model = load_model(model_path)

labels = {
    0: 'Corn___Common_rust',
    1: 'Corn___Northern_Leaf_Blight',
    2: 'Corn___healthy',
    3: 'Not_a_Leaf',
    4: 'Potato___Early_blight',
    5: 'Potato___Late_blight',
    6: 'Potato___healthy',
    7: 'Strawberry___Leaf_scorch',
    8: 'Strawberry___healthy',
}

# Dictionary to store solution messages for each class
solution_messages = {
    'Corn___Common_rust': 'Utilize resistant hybrids, fungicides, rotation, and residue destruction',
    'Corn___Northern_Leaf_Blight': 'Opt for resistant hybrids, rotate crops, and use fungicides',
    'Corn___healthy': 'No action needed. Plant is healthy',
    'Not_a_Leaf': 'Please Provide the  Plant Image',
    'Potato___Early_blight': 'Choose resistant varieties, avoid overhead irrigation, rotate crops, and balance nutrients',
    'Potato___Late_blight': 'Choose resistant varieties, avoid overhead watering,dispose of plant debris away from growing areas',
    'Potato___healthy': 'No action needed. Plant is healthy.',
    'Strawberry___Leaf_scorch': 'Use resistant varieties, remove infected leaves, ensure proper air circulation, and avoid overhead irrigation.',
    'Strawberry___healthy': 'No action needed. Plant is healthy.'
}
reason_messages = {
    'Corn___Common_rust': 'Fungus Puccinia sorghi',
    'Corn___Northern_Leaf_Blight': 'Fungus Exserohilum turcicum.',
    'Corn___healthy': 'No issue, It is healthy',
    'Not_a_Leaf': 'Please Provide the  Plant Image.',
    'Potato___Early_blight': 'Fungus Alternaria solani.',
    'Potato___Late_blight': 'Oomycete pathogen Phytophthora infestans.',
    'Potato___healthy': 'No issue, It is healthy',
    'Strawberry___Leaf_scorch': 'Fungus Diplocarpon earliana.',
    'Strawberry___healthy': 'No issue, It is healthy',
}

# Function to extract reason message based on predicted class name
def get_reason_message(predicted_class_name):
    return reason_messages.get(predicted_class_name, None)

# Function to extract solution message based on predicted class name
def get_solution_message(predicted_class_name):
    return solution_messages.get(predicted_class_name, None)

current_working_directory = os.getcwd()
temp_dir = os.path.join(current_working_directory, 'temp_images')

if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['img_file']
        if file:
            try:
                img_url = save_and_process_image(file)
                result = process_image(img_url)
                return render_template('index.html', result=result, img_url=img_url)
            except Exception as e:
                error_message = f"Error processing image: {str(e)}"
                print(f"Error: {error_message}")
                return render_template('index.html', result=None, img_url=None, error_message=error_message)

    return render_template('index.html', result=None, img_url=None)

def save_and_process_image(file):
    img_path = os.path.join(temp_dir, f"temp_{file.filename}")
    file.save(img_path)
    return img_path

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['img_file']
        if file:
            img_url = save_and_process_image(file)
            result = process_image(img_url)
            return jsonify(result)
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Error: {error_message}")
        return jsonify({'error': error_message})

    return jsonify({'error': 'Invalid request'})

def process_image(img_url):
    try:
        specific_image = image.load_img(img_url, target_size=(IMG_WIDTH, IMG_HEIGHT))
        specific_image_array = image.img_to_array(specific_image)
        specific_image_array = np.expand_dims(specific_image_array, axis=0)
        specific_image_array /= 255.0

        print("Shape of input image array:", specific_image_array.shape)

        prediction = loaded_model.predict(specific_image_array)
        print("Raw prediction values:", prediction)

        
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = labels[predicted_class_index]
        
  
        reason_message = get_reason_message(predicted_class_name)
        solution_message = get_solution_message(predicted_class_name)

        result = {
        'actual_class_name': None,
        'predicted_class_name': predicted_class_name,
        'reason_message': reason_message,
        'solution_message': solution_message
        }

        print("Predicted class index:", predicted_class_index)
        print("Predicted class name:", predicted_class_name)
        print("Reason message:", reason_message)
        print("solution message: ",solution_message)


        return result

        
    except Exception as e:
        error_message = f"Error processing image: {str(e)}"
        print(f"Error: {error_message}")
        return {'error': error_message}

if __name__ == '__main__':
    app.run(debug=True)
