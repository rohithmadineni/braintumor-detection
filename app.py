import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# Load data
def load_data(data_dir):
    images = []
    labels = []
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    for label in ["tumor", "no_tumor"]:
        label_dir = os.path.join(data_dir, label)
        for file in os.listdir(label_dir):
            if not file.lower().endswith(valid_extensions):
                continue  # Skip non-image files
            file_path = os.path.join(label_dir, file)
            image = Image.open(file_path).resize((128, 128))
            # Convert image to RGB if it's not
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image) / 255.0
            images.append(image)
            labels.append(1 if label == "tumor" else 0)
    return np.array(images), np.array(labels)

# Set up data directory
data_dir = "brain_tumor_dataset(1)"  # Replace with the path to your dataset
X, y = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Transfer Learning with MobileNetV2
base_model = MobileNetV2(include_top=False, input_shape=(128, 128, 3), weights='imagenet')
base_model.trainable = False  # Freeze the base model initially

# Define the model with explicit input shape
model = Sequential([
    Input(shape=(128, 128, 3)),
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Check input shape of the model
print("Model input shape:", model.input_shape)

optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Save the model during training in the new Keras format (.keras)
checkpoint = ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=64),
    epochs=50,
    validation_data=(X_test, y_test),
    callbacks=[reduce_lr, early_stopping, checkpoint]
)

# After training, manually save the model in .h5 format
model.save('final_model.h5')  # This will save the model in .h5 format


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Save the accuracy to a file
with open('accuracy.txt', 'w') as f:
    f.write(str(accuracy))

# Flask App
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
    # Convert image to RGB if it's not
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
