import numpy  as np
import pandas as pd

from copy       import copy
from customDefs import ECO_SEED
from customDefs import readELSA, getDomains
from customDefs import plotPanelPerf, LRModel

from sklearn.model_selection import train_test_split

def main():

  # defines the application parameters
  np.random.seed(ECO_SEED)
  targetpath = ['C:\\', 'Users', 'Andre', 'Task Stage', 'Task - collabSergio', 'collabSergio', 'datasets']
  filename   = 'DATASETELSA.tsv'

  # loads the dataset
  df = readELSA(targetpath, filename)
  #df.info()
  domains = getDomains(['eGFR', 'eDCE', 'eACR', 'eBSA', 'SCr', 'UCr', 'UAlb', 'height', 'weight'], df)

  # partitions the dataset into ckd+ and ckd-; also splits ckd- into training and test partitions
  ckdpos = copy(df.loc[(df['hasCKD'] >  0)]) # dataframe com os sujeitos DRC-positivos
  ckdneg = copy(df.loc[(df['hasCKD'] == 0)]) # dataframe com os sujeitos DRC-negativos
  ckdneg_tr, ckdneg_te = train_test_split(ckdneg, test_size=0.33)

  ss = len(df)
  ss_ckdneg = len(ckdneg)
  ss_ckdpos = len(ckdpos)
  prevalence = (ss_ckdpos/ss) * 100

  print('Número de indivíduos na amostra .......... .............: {0:5d}'.format(ss))
  print('Número de indivíduos portadores de DRC .................: {0:5d} (prevalência {1:4.1f}%)'.format(ss_ckdpos, prevalence))
  print('Número de indivíduos não portadores de DRC .............: {0:5d}'.format(ss_ckdneg))
  print('Número de indivíduos na partição de treinamento ........: {0:5d}'.format(len(ckdneg_tr)))
  print('Número de indivíduos na partição de teste ..... ........: {0:5d}'.format(len(ckdneg_te)))

  # plots the relationship between Age and other variables
  plotTitle = 'Relações entre Age e outras variáveis'
  (outcome, predVars) = ('Age', ['eGFR', 'eDCE', 'eACR', 'eBSA', 'SCr', 'UCr', 'UAlb'])
  plotPanelPerf(outcome, predVars, ckdneg_te, ckdneg_te[outcome], ckdpos, ckdpos[outcome], plotTitle, domains, legend = False)

  # fits the data to a LR model and evaluates it
  plotTitle = 'Modelo 1: Regressão Linear usando variáveis da equação CKD-EPI'
  outcome   = 'Age'
  predVars  = ['Gender'] #['eDCE', 'SCr', 'Gender', 'Race']
  results = LRModel(outcome, predVars, ckdneg_tr, ckdneg_te, ckdpos, plotTitle, domains)

if __name__ == "__main__":

  main()
