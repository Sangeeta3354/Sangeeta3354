import numpy
from sklearn import linear_model

a = numpy.array([3.78, 2.44, 2.09, 0.14, 1.72, 1.65, 4.92, 4.37, 4.96, 4.52, 3.69, 5.88]).reshape(-1,1)
b = numpy.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

logR = linear_model.LogisticRegression()
logR.fit(a,b)

def logit2prob(logR, a):
  log_odds = logR.coef_ * a + logR.intercept_
  odds = numpy.exp(log_odds)
  probability = odds / (1 + odds)
  return(probability)

print(logit2prob(logR, a))