import numpy  as np
import pandas as pd

from copy        import copy
from scipy.stats import kendalltau
from customDefs  import ECO_SEED
from customDefs  import readELSA, saveAsText, serialise, deserialise
from customDefs  import SRT, LRModel, getDomains, plotPanelPerf, headerfy

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler

def main():

  # defines the application parameters
  np.random.seed(ECO_SEED)
  targetpath = ['C:\\', 'Users', 'Andre', 'Task Stage', 'Task - collabSergio', 'collabSergio', 'datasets']
  filename   = 'DATASETELSA.tsv'

  # loads the dataset
  print('Loading the ELSA dataset')
  #df = readELSA(targetpath, filename)
  df = deserialise('elsa')
  print(df.info())
  serialise(df, 'elsa')

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

  #
  print('Preparing partitions')
  outcome  = 'Age'
  exclude = ['IDElsa', 'eGFR', 'hasCKD']
  predVars = sorted(set(df.columns).difference([outcome] + exclude))

  buffer = '{0}\t{1}\t{2}\t{3:5.3f}\t{4:5.3f}\t{5}\t{6:5.3f}\t{7:5.3f}\t{8:5.3f}\t{9:5.3f}\t{10:5.3f}'
  header = headerfy(buffer).format('Dataset', 'Base', 'field', 'min', 'median', 'mean', 'max', '#dom', 'tau', 'p-value', '|tau|')
  content = [header]

  for (basedslbl, baseds) in [('whole', df), ('CKD-', ckdneg), ('CKD+', ckdpos)]:

    rawds  = df[predVars].to_numpy(copy = True)
    scaler = StandardScaler().fit(baseds[predVars].to_numpy())
    stdds  = scaler.transform(rawds)
    datasources = [('raw', rawds, predVars), ('standardised', stdds, predVars)]

    for ndims in range(2, len(predVars)):
      (_, _, params_srt) = SRT(baseds, predVars, outcome, activation = 'identity', ndims=ndims)
      (srtds, _, _)      = SRT(df,     predVars, outcome, params=params_srt)
      datasources.append(('SRTed({0})'.format(ndims), srtds, None))

    X = df[outcome].to_numpy()
    for (dslbl, ds, fields) in datasources:
      print('-- processing the {0} dataset.'.format(dslbl + '.' + basedslbl))
      if(fields is None):
        fields = ['field_{0}'.format(i+1) for i in range(ds.shape[1])]
      for i in range(len(fields)):
        Y = ds[:, i]
        (tau, pvalue) = kendalltau(X, Y)
        content.append((buffer.format(dslbl, basedslbl, fields[i], Y.min(), np.median(Y), Y.mean(), Y.max(), len(set(Y)), tau, pvalue, abs(tau))))

  saveAsText('\n'.join(content), 'selection.csv')

  return None

if __name__ == "__main__":

  main()
