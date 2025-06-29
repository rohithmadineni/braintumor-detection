import os
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Create Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
model = load_model('final_model.h5')  # Load the saved model

def load_accuracy():
    with open('accuracy.txt', 'r') as f:
        accuracy = float(f.read())
    return accuracy

accuracy = load_accuracy()

def predict(image_path):
    image = Image.open(image_path).resize((128, 128))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)
    return "Tumor Detected" if prediction[0][0] > 0.5 else "No Tumor Detected"

@app.route("/", methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            result = predict(file_path)
            return render_template("result.html", result=result, accuracy=accuracy, image_url=filename)
    return render_template("upload.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
