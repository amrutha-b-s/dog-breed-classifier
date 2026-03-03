import os
from flask import Flask, render_template, request, send_from_directory, redirect
from PIL import Image
import numpy as np

app = Flask(__name__)

# ---------------- HOME ----------------
@app.route("/")
def home():
    return render_template("index.html")


# ---------------- PREDICT ----------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        file = request.files.get("file")

        if not file or file.filename == "":
            return "Please upload an image."

        # Safe image processing
        img = Image.open(file.stream).convert("RGB")
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Demo prediction (since no TensorFlow on Render free)
        prediction_text = "Prediction Successful (Demo Mode)"

        return render_template("result.html", prediction=prediction_text)

    except Exception as e:
        return f"Error: {str(e)}"


# ---------------- PDF ROUTES ----------------
@app.route("/read")
def read_pdf():
    return send_from_directory("static", "report.pdf")

@app.route("/download")
def download_pdf():
    return send_from_directory("static", "report.pdf", as_attachment=True)

@app.route("/skip")
def skip():
    return redirect("/")


# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(debug=True)