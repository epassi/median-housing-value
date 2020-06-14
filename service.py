import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS, cross_origin
# from sklearn.externals import joblib
import joblib

app = Flask(__name__)
# app.config['CORS_HEADERS'] = 'Content-Type'

# CORS(app, resources={r'/*': {'origins': '*'}})



# model = joblib.load(open('median-house-value.pkl', 'rb'))
model = joblib.load('median-house-value.joblib')

@app.route('/median-house-value',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

if __name__ == "__main__":
    app.run(debug=True)
