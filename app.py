from flask import Flask, request, render_template
import os
import numpy as np
from PIL import Image

app = Flask(__name__)

# ---------------------------------------------------
# Detect if running on Render
# ---------------------------------------------------
ON_RENDER = os.getenv("RENDER") is not None

# ---------------------------------------------------
# Load model only if NOT on Render
# ---------------------------------------------------
if not ON_RENDER:
    import tensorflow as tf
    model = tf.keras.models.load_model("model.h5")
else:
    model = None

# ---------------------------------------------------
# Home Route
# ---------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html")

# ---------------------------------------------------
# Prediction Route
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded"

    file = request.files["file"]

    if file.filename == "":
        return "No selected file"

    # If deployed version → skip real prediction
    if ON_RENDER or model is None:
        return render_template("result.html",
                               prediction="Model disabled in deployed version")

    # ---------- Local Prediction ----------
    img = Image.open(file).resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    return render_template("result.html",
                           prediction=f"Predicted Class: {predicted_class}")

# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)