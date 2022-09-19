from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib


app = Flask(__name__)
custReviewmodel=pickle.load(open('Customer_Reviews_sentimentanalisysmodel.pkl','rb'))

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	clf = joblib.load(custReviewmodel)

	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		my_prediction = clf.predict(data)
	return render_template('result.html',prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug=True)