import pickle
import flask
from flask import Flask, request, app, jsonify, url_for, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)
model=pickle.load(open('airfoil_model_regression.pkl','rb'))

@app.route('/')
def home():
    return render_template("home.html")


@app.route('/predict_postman', methods=["POST"])
def predict_postman():
    data=request.json["data"]
    print(data)
    data_array=[list(data.values())]
    output_data_array = model.predict(data_array)[0]
    return jsonify(output_data_array)


def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output = model.predict(final_features)[0]
    print(output)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))


if __name__=="__main__":
     app.run(debug=True)
