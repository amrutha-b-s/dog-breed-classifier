import os
from flask import Flask, render_template, request, send_from_directory, redirect

app = Flask(__name__)

# Ensure static folder exists
if not os.path.exists("static"):
    os.makedirs("static")


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

        # Save uploaded image inside static folder
        upload_path = os.path.join("static", file.filename)
        file.save(upload_path)

        # ---- DEMO PREDICTION (for Render Free Plan) ----
        # Replace this section with real model logic in local version

        breed = "Doberman"
        confidence = "85.56"

        return render_template(
            "result.html",
            breed=breed,
            confidence=confidence,
            image_url="/" + upload_path
        )

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