from flask import Flask, render_template, request
from sklearn import preprocessing
import numpy as np
import pickle

app = Flask(__name__)

pickle_in = open("model.pickle","rb")
model = pickle.load(pickle_in)



@app.route('/')
def home_page():
	return render_template('predictorform.html')
@app.route('/results', methods=['POST'])
def results():
	form = request.form
	if request.method == 'POST':
		RPM = request.form.get('RPM')
		STOR = request.form.get('STOR')
		HKLX = request.form.get('HKLX')
		HOOKLD = request.form.get('HOOKLD')
		HKLD = request.form.get('HKLD')
		SPPA1 = request.form.get('SPPA1')
		SPPA2 = request.form.get('SPPA2')
		SPPA = request.form.get('SPPA')
		TQX = request.form.get('TQX')
		array = np.array([[RPM,STOR,HKLX,HOOKLD,HKLD,SPPA1,SPPA2,SPPA,TQX]])
		n_array = preprocessing.normalize(array)
		predicted_value = model.predict(n_array)
			
		if predicted_value == 1: predicted_value = "Ceci est une situation de stuck-pipe."
		else: predicted_value = "Ceci n'est pas une situation de stuck-pipe."
	

		return render_template('resultsform.html',RPM=RPM,STOR=STOR,
			HKLX=HKLX,HOOKLD=HOOKLD,HKLD=HKLD,SPPA1=SPPA1,SPPA2=SPPA2,SPPA=SPPA,TQX=TQX ,predicted_result=predicted_value)

if __name__ == '__main__':
	app.run(debug='true')
