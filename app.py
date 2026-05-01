import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from inference.predict_single_image import predict_image

# Upload folder inside static
UPLOAD_FOLDER = os.path.join("static", "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Create folder if not exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    filename = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", prediction=None, filename=None)

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", prediction=None, filename=None)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)

            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Only get label (ignore confidence)
            label, _ = predict_image(filepath)

            prediction = label

    return render_template(
        "index.html",
        prediction=prediction,
        filename=filename
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)