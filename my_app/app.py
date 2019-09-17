from flask import Flask, render_template, request
import numpy as np
from sklearn.externals import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')





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
		predicted_stock_price = model.predict_proba(array)
		return render_template('resultsform.html',RPM=RPM,STOR=STOR,
			HKLX=HKLX,HOOKLD=HOOKLD,HKLD=HKLD,SPPA1=SPPA1,SPPA2=SPPA2,SPPA=SPPA,TQX=TQX ,predicted_price=predicted_stock_price)

if __name__ == '__main__':
	app.run(debug='true')