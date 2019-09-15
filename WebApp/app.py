from flask import Flask, render_template, request, jsonify
import numpy as np
from sklearn.externals import joblib


app = Flask(__name__) 
model = joblib.load('RF_model.pkl')




@app.route('/')
def index():
	return render_template('home.html')

@app.route('/api',methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([[np.array(data['RPM','STOR','HKLX','HOOKLD','HKLD','SPPA1','SPPA2','SPPA','TQX'])]])
    output = prediction[0]
    return jsonify(output)

if __name__ == '__main__':
	app.run(debug='true')