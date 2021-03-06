import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)
model = joblib.load('median-house-value.joblib')

@app.route('/', methods=['GET'])
def root():
    return 'hey there'

@app.route('/median-house-value',methods=['POST'])
def results():
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
