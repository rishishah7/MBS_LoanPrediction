from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
import joblib
from techno_modular import test_model

# load the model from disk

app = Flask(__name__,template_folder='templates')

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		int_features = [int(x) for x in request.form.values()]
		final_features = np.array(int_features)
		my_prediction = test_model(final_features)
		print(my_prediction)
	return render_template("index.html",prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)