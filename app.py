import os
from datetime import datetime

from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

from inference.predict_single_image import predict_image

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

UPLOAD_FOLDER = os.path.join(
    "static",
    "uploads"
)

ALLOWED_EXTENSIONS = {
    "jpg",
    "jpeg",
    "png"
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app = Flask(__name__)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# --------------------------------------------------
# CHECK FILE
# --------------------------------------------------

def allowed_file(filename):

    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() \
           in ALLOWED_EXTENSIONS


# --------------------------------------------------
# ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])

def index():

    prediction = None
    filename = None
    timestamp = None

    if request.method == "POST":

        if "image" not in request.files:

            return render_template(
                "index.html",
                prediction=None
            )

        file = request.files["image"]

        if file.filename == "":

            return render_template(
                "index.html",
                prediction=None
            )

        if file and allowed_file(file.filename):

            filename = secure_filename(
                file.filename
            )

            filepath = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(filepath)

            # AI Prediction
            prediction = predict_image(filepath)

            timestamp = datetime.now().strftime(
                "%d %B %Y | %I:%M %p"
            )

    return render_template(
        "index.html",
        prediction=prediction,
        filename=filename,
        timestamp=timestamp
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=True
    )