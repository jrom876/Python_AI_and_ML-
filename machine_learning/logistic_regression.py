## https://www.w3schools.com/python/python_ml_logistic_regression.asp

import numpy
from sklearn import linear_model

X = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
y = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

def logit2prob(logr,x):
  log_odds = logr.coef_ * x + logr.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

if __name__ == '__main__':  	

	logr = linear_model.LogisticRegression()
	logr.fit(X,y)

	#predict if tumor is cancerous where the size is 3.46mm:
	predicted = logr.predict(numpy.array([3.96]).reshape(-1,1))
	log_odds = logr.coef_
	odds = numpy.exp(log_odds)

	print("odds = {0}".format(odds)) 
	print(predicted)
	print(logit2prob(logr, X)) 

