import sys
import numpy  as np
import pandas as pd

from season3     import load_ELSA, load_PNS2013lab
from scipy.stats import bootstrap
from sharedDefs  import headerfy

def prot1(dataset, n_neighbors, tries):

  ECO_SEED = 31
  np.random.seed(ECO_SEED)
  np.set_printoptions(precision=5, suppress=True, linewidth=3000)

  print()
  if dataset.lower() == 'elsa':
    print('Loading and preprocessing the ELSA dataset')
    ds = load_ELSA()
  elif(dataset.lower() == 'pns'):
    print('Loading and preprocessing the PNS 2013 lab exams dataset')
    ds = load_PNS2013lab()
  #print(ds.info())

  # splits the dataset into two partitions:
  # -- pep: dedicated to estimate parameters from Campbell's kidney age equation
  # --  ds: used to all remaining purposes
  mask = '{0}\t{1}\t{2:4.2f}\t{3:4d}\t{4:5.1f}\t{5:5.1f}\t{6:5.1f}'
  header = headerfy(mask).format('ds', 'gr', 'frac', 'ns', 'mean', 'low', 'high')
  print(header)
  for pep_frac in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    #pep_frac = 0.50
    pep_size = int(pep_frac * len(ds))
    ids = ds['ID'].tolist()
    np.random.shuffle(ids)
    pep = ds.loc[ds['ID'].isin(ids[:pep_size])]
    #dso = ds.loc[ds['ID'].isin(ids[pep_size:])]

    # collects samples for younger (g1) and older (g2) age groups
    g1 = pep['eGFR'].loc[(pep['hasCKD'] == 0) & (pep['Age'] >  0) & (pep['Age'] <= 35)].to_numpy()
    g2 = pep['eGFR'].loc[(pep['hasCKD'] == 0) & (pep['Age'] > 35) & (pep['Age'] <= 75)].to_numpy()
    g2 = (g2 - g1.mean())/(75-35)

    # estimates the parameters from Campbell's kidney age model
    ci_g1 = bootstrap([g1,], np.mean, method='BCa').confidence_interval
    ci_g2 = bootstrap([g2,], np.mean, method='BCa').confidence_interval

    #
    buffer_g1 = mask.format(dataset, 'g1', pep_frac, len(g1), g1.mean(), ci_g1.low, ci_g1.high)
    buffer_g2 = mask.format(dataset, 'g2', pep_frac, len(g2), g2.mean(), ci_g2.low, ci_g2.high)
    print(buffer_g1)
    print(buffer_g2)

def main(dataset, n_neighbors, tries):

  ECO_SEED = 31
  np.random.seed(ECO_SEED)
  np.set_printoptions(precision=5, suppress=True, linewidth=3000)

  print()
  if dataset.lower() == 'elsa':
    print('Loading and preprocessing the ELSA dataset')
    ds = load_ELSA()
  elif(dataset.lower() == 'pns'):
    print('Loading and preprocessing the PNS 2013 lab exams dataset')
    ds = load_PNS2013lab()
  #print(ds.info())

  # splits the dataset into two partitions:
  # -- pep: dedicated to estimate parameters from Campbell's kidney age equation
  # --  ds: used to all remaining purposes
  mask = '{0}\t{1}\t{2:4.2f}\t{3:4d}\t{4:5.1f}\t{5:5.1f}\t{6:5.1f}'
  header = headerfy(mask).format('ds', 'gr', 'frac', 'ns', 'mean', 'low', 'high')
  print(header)
  pep_frac = 0.3
  pep_size = int(pep_frac * len(ds))
  ids = ds['ID'].tolist()
  np.random.shuffle(ids)
  pep = ds.loc[ds['ID'].isin(ids[:pep_size])]
  dso = ds.loc[ds['ID'].isin(ids[pep_size:])]

  # collects samples for younger (g1) and older (g2) age groups
  g1 = pep['eGFR'].loc[(pep['hasCKD'] == 0) & (pep['Age'] >  0) & (pep['Age'] <= 35)].to_numpy()
  g2 = pep['eGFR'].loc[(pep['hasCKD'] == 0) & (pep['Age'] > 35) & (pep['Age'] <= 75)].to_numpy()
  g2 = (g2 - g1.mean())/(75-35)

  # estimates the parameters from Campbell's kidney age model
  ci_g1 = bootstrap([g1,], np.mean, method='BCa').confidence_interval
  ci_g2 = bootstrap([g2,], np.mean, method='BCa').confidence_interval

  #
  buffer_g1 = mask.format(dataset, 'g1', pep_frac, len(g1), g1.mean(), ci_g1.low, ci_g1.high)
  buffer_g2 = mask.format(dataset, 'g2', pep_frac, len(g2), g2.mean(), ci_g2.low, ci_g2.high)
  print(buffer_g1)
  print(buffer_g2)

  #
  dso = dso.copy()
  _gfr_young   = g1.mean()
  _gfr_decline = g2.mean()
  dso['kAge'] = dso.apply(lambda row: (1/_gfr_decline) * (_gfr_young - row.eGFR) + 40.0, axis=1)
  dso['KCD']  = dso.apply(lambda row: row.kAge - row.Age, axis=1)
  cols = ['ID', 'Age', 'eGFR', 'hasCKD', 'kAge', 'KCD']
  dso[cols].to_csv('dso.csv')

if(__name__ == '__main__'):

  dataset = sys.argv[1]
  n_neighbors = int(sys.argv[2])
  tries = int(sys.argv[3])
  main(dataset, n_neighbors, tries)
