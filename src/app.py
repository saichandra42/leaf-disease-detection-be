from flask import request,send_file,Flask,jsonify
from flask_cors import CORS, cross_origin
import os
import json
from model import cnn_model_prediction

app = Flask(__name__)
cwd = os.getcwd()
app.config['UPLOAD_FOLDER'] = './upload'

CORS(app)

@app.route("/",methods=['GET'])
def index():
    return "Working"


@app.route("/uploadImage",methods = ["POST"])
def uploadImage():
    prediction = ''
    if request.method == "POST":
        if request.files:
            image = request.files['file']
            basedir = os.path.abspath(os.path.dirname(__file__))
            path = os.path.join(basedir, app.config['UPLOAD_FOLDER'], image.filename)
            image.save(path)
            prediction = cnn_model_prediction(path)
    return prediction

if __name__ == "__main__":
    app.run(port=5000,debug=True)