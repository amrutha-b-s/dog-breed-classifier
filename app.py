import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load pretrained model
model = MobileNetV2(weights="imagenet")

CONFIDENCE_THRESHOLD = 0.20  # Lowered for demo accuracy

UPLOAD_FOLDER = os.path.join("static", "uploads")
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    if file.filename == "":
        return "No file selected"

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # Correct input size
    img = image.load_img(filepath, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    preds = model.predict(img_array)
    decoded = decode_predictions(preds, top=1)[0][0]

    breed = decoded[1]
    confidence = float(decoded[2])

    if confidence < CONFIDENCE_THRESHOLD:
        breed = "Breed not recognized"

    confidence_percent = round(confidence * 100, 2)

    return render_template(
        "result.html",
        image_path="uploads/" + filename,
        breed=breed,
        confidence=confidence_percent
    )

@app.route("/read_pdf")
def read_pdf():
    return send_from_directory("static", "report.pdf")

@app.route("/download_pdf")
def download_pdf():
    return send_from_directory("static", "report.pdf", as_attachment=True)

if __name__ == "__main__":
    app.run(debug=True)