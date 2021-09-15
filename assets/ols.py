import pandas as pd
import numpy  as np
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from scipy import stats

# https://stackoverflow.com/questions/27928275/find-p-value-significance-in-scikit-learn-linearregression
def version1():

  diabetes = datasets.load_diabetes()
  X = diabetes.data
  y = diabetes.target

  X2 = sm.add_constant(X)
  est = sm.OLS(y, X2)
  est2 = est.fit()
  print(est2.summary())

  input('Press any key to continue ...')

  lm = LinearRegression()
  lm.fit(X,y)
  params = np.append(lm.intercept_,lm.coef_)
  predictions = lm.predict(X)

  newX = np.append(np.ones((len(X),1)), X, axis=1)
  MSE = (sum((y-predictions)**2))/(len(newX)-len(newX[0]))

  var_b = MSE*(np.linalg.inv(np.dot(newX.T,newX)).diagonal())
  sd_b = np.sqrt(var_b)
  ts_b = params/ sd_b

  p_values =[2*(1-stats.t.cdf(np.abs(i),(len(newX)-len(newX[0])))) for i in ts_b]

  sd_b = np.round(sd_b,3)
  ts_b = np.round(ts_b,3)
  p_values = np.round(p_values,3)
  params = np.round(params,4)

  myDF3 = pd.DataFrame()
  myDF3["Coefficients"],myDF3["Standard Errors"],myDF3["t values"],myDF3["Probabilities"] = [params,sd_b,ts_b,p_values]
  print(myDF3)

def version2():

  diabetes = datasets.load_diabetes()
  X = diabetes.data
  y = diabetes.target

  X2 = sm.add_constant(X)
  est = sm.OLS(y, X2)
  est2 = est.fit()
  print(est2.summary())

def version3():

  diabetes = datasets.load_diabetes()
  X = diabetes.data
  y = diabetes.target

  model = sm.OLS(y, sm.add_constant(X))
  results = model.fit()
  #print(results.summary())
  print(results.params)

def version4():

  duncan_prestige = sm.datasets.get_rdataset("Duncan", "carData")
  Y = [y for y in duncan_prestige.data['income']]
  X = [x for x in duncan_prestige.data['education']]

  model = sm.OLS(Y, sm.add_constant(X))
  results = model.fit()
  (beta, alpha) = results.params
  print(alpha, beta)

def main():

  #version1()

  #version2()

  version3()

  #version4()

if __name__ == "__main__":

  main()
