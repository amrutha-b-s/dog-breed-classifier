import os
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# load trained model
model = load_model("dog_breed_model.keras")

# IMPORTANT: must match training order
class_names = [
    "afghan_hound",
    "beagle",
    "bernese_mountain_dog",
    "maltese_dog",
    "pomeranian",
    "samoyed"
]

# -------------------------
# prediction function
# -------------------------
def predict_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)

    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    confidence = float(np.max(predictions)) * 100

    breed = class_names[predicted_index]

    return breed, confidence


# -------------------------
# home page
# -------------------------
@app.route("/")
def index():
    return render_template("index.html")


# -------------------------
# prediction route
# -------------------------
@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    breed, confidence = predict_image(filepath)

    return render_template(
        "result.html",
        breed=breed,
        confidence=round(confidence, 2),
        image_file=file.filename
    )


# -------------------------
# read pdf
# -------------------------
@app.route("/read_pdf")
def read_pdf():
    return send_from_directory("static", "report.pdf")


# -------------------------
# download pdf
# -------------------------
@app.route("/download_pdf")
def download_pdf():
    return send_from_directory("static", "report.pdf", as_attachment=True)


# -------------------------
# run app
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)