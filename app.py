from flask import Flask, render_template, request, send_file
from io import BytesIO

# Import the functions
from example import load_flux_pipeline, generate_image

app = Flask(__name__)

# Load the pipeline once
pipe = load_flux_pipeline()

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        prompt = request.form["prompt"]

        # Generate image from the prompt
        image = generate_image(pipe, prompt)

        # Save image to a byte stream
        img_io = BytesIO()
        image.save(img_io, 'PNG')
        img_io.seek(0)

        return send_file(img_io, mimetype='image/png')

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
