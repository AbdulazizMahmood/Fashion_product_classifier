from flask import Flask, render_template, request, url_for, send_from_directory
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
import numpy as np
import os
from werkzeug.utils import secure_filename
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__, static_folder='uploads')
app.config['UPLOAD_FOLDER'] = 'uploads'

model = tf.keras.models.load_model('fashion_model.keras')



class_names = ['gender_Boys', 'gender_Girls', 'gender_Men', 'gender_Unisex',
       'gender_Women', 'baseColour_Beige', 'baseColour_Black',
       'baseColour_Blue', 'baseColour_Bronze', 'baseColour_Brown',
       'baseColour_Burgundy', 'baseColour_Charcoal', 'baseColour_Coffee Brown',
       'baseColour_Copper', 'baseColour_Cream', 'baseColour_Fluorescent Green',
       'baseColour_Gold', 'baseColour_Green', 'baseColour_Grey',
       'baseColour_Grey Melange', 'baseColour_Khaki', 'baseColour_Lavender',
       'baseColour_Lime Green', 'baseColour_Magenta', 'baseColour_Maroon',
       'baseColour_Mauve', 'baseColour_Metallic', 'baseColour_Multi',
       'baseColour_Mushroom Brown', 'baseColour_Mustard',
       'baseColour_Navy Blue', 'baseColour_Nude', 'baseColour_Off White',
       'baseColour_Olive', 'baseColour_Orange', 'baseColour_Peach',
       'baseColour_Pink', 'baseColour_Purple', 'baseColour_Red',
       'baseColour_Rose', 'baseColour_Rust', 'baseColour_Sea Green',
       'baseColour_Silver', 'baseColour_Skin', 'baseColour_Steel',
       'baseColour_Tan', 'baseColour_Taupe', 'baseColour_Teal',
       'baseColour_Turquoise Blue', 'baseColour_White', 'baseColour_Yellow']

def predict_top_labels(model, image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = efficientnet_preprocess(image_array)

    predictions = model.predict(image_array)[0]
    gender_predictions = [(label, predictions[i])
                          for i, label in enumerate(class_names)
                          if label.startswith("gender_")]
    top_gender = max(gender_predictions, key=lambda x: x[1]) if gender_predictions else None

    colour_predictions = [(label, predictions[i])
                          for i, label in enumerate(class_names)
                          if label.startswith("baseColour_")]
    top_colour = max(colour_predictions, key=lambda x: x[1]) if colour_predictions else None
    
    return top_gender, top_colour

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route("/", methods=["GET", "POST"])
def classify_image():
    image_url = None
    predicted_gender = None
    predicted_color = None
    
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part in the request"
        file = request.files["file"]
        if file.filename == "":
            return "No file selected"
            
        if file:
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                gender_pred, color_pred = predict_top_labels(model, filepath)
                
                predicted_gender = gender_pred[0].replace('gender_', '') if gender_pred else None
                predicted_color = color_pred[0].replace('baseColour_', '') if color_pred else None
                
                image_url = url_for('uploaded_file', filename=filename)
                
            except Exception as e:
                return f"Error processing image: {str(e)}"

    return render_template("index.html", 
                         predicted_gender=predicted_gender, 
                         predicted_color=predicted_color, 
                         image_url=image_url)

if __name__ == "__main__":
    # Create upload folder if it doesn't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
