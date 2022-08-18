# calculate the kendall's correlation between two variables
import sys
from numpy.random import rand
from numpy.random import seed
from scipy.stats  import kendalltau

# seed random number generator
seed(23)
alpha = 0.05

def assess(coef, p):

  print('\nKendall correlation coefficient: %.3f' % coef)

  # interpret the significance
  if p > alpha:
    print('Samples are uncorrelated (fail to reject H0) p=%.3f' % p)
  else:
    print('Samples are correlated (reject H0) p=%.3f' % p)

  return None

def rank(data, reverse = False):
  L = sorted(set(data))
  if(reverse):
    L.reverse()
  return [(1 + L.index(v)) for v in data]

def main(ns):
	
  # prepare data
  data1 = rand(ns) * 20
  data2 = data1 + (rand(ns) * 10)

  # calculate kendall's correlation (from the raw data)
  coef1, p1 = kendalltau(data1, data2)
  assess(coef1, p1)

  data1t = rank(data1)
  data2t = rank(data2)
  coef2, p2 = kendalltau(data1t, data2t)
  assess(coef2, p2)

if __name__ == "__main__":
  ns = int(sys.argv[1])
  main(ns)

