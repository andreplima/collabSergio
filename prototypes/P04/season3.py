# next task - make loaders as similar as possible
import sys
import codecs
import pickle
import numpy as np
import pandas as pd

from os.path     import exists
from collections import defaultdict
from scipy.stats import bootstrap

from sklearn.neighbors      import NearestNeighbors
from sklearn.linear_model   import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing  import StandardScaler

import warnings
warnings.filterwarnings('error')

def serialise(obj, name):
  f = open(name + '.pkl', 'wb')
  p = pickle.Pickler(f)
  p.fast = True
  p.dump(obj)
  f.close()
  p.clear_memo()

def deserialise(name):
  f = open(name + '.pkl', 'rb')
  p = pickle.Unpickler(f)
  obj = p.load()
  f.close()
  return obj

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

def estimate_GFR_CKDEPI(scr, age, gender, race):
  """
  Estimates the GFR of an individual using the CKD-epi equation

  Parameters:
    scr is serum creatinine measured in mg/dL using
    -- (PREENCHER tipo de assay, see https://youtu.be/KcoeVtiYGpU)
    age in years, gender and race as strings in the domains described above and encoded as:
    -- gender: female is encoded as 1, and male as 0
    -- race: white is encoded as 0, and non-whites (including blacks) as 1
  Returns
    estimated GFR in mL/min/1.73m^2
    -- see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763564/, page 14 (see table footnote)

  Divergência com valores calculados por https://www.sbn.org.br/profissional/utilidades/calculadoras-nefrologicas/ ...
  ... se deve ao fato desta usar as 8 equações individuais, nas quais o primeiro coeficiente é arredondado

  """

  kappa = lambda gender:  0.7   if gender == 1 else  0.9
  alpha = lambda gender: -0.329 if gender == 1 else -0.411
  scr_k = scr/kappa(gender)

  eGFR = (141.0
          * (min(scr_k, 1.0) ** alpha(gender))
          * (max(scr_k, 1.0) ** -1.209)
          * (0.993) ** age
          * (1.018 if gender == 1 else 1.0)
          * (1.159 if race   == 1 else 1.0)
         )

  return eGFR

def estimate_GFR_DCE(ucr, scr, uvol, udur, height, weight):
  # parameters:
  #   ucr is urine creatinine measured in mg/dL using (PREENCHER tipo de assay, see https://youtu.be/KcoeVtiYGpU)
  #   scr is serum creatinine measured in mg/dL using (PREENCHER tipo de assay, see https://youtu.be/KcoeVtiYGpU)
  #   uvol é o volume coletado de urina, em mL
  #   udur é a duração da coleta de urina, em minutos (valores típicos em torno de 660 a 720)
  #   height é a altura, em cm
  #   weight é o peso em kG
  #   -- returns estimated GFR in mL/min/1.73m^2
  # estimador descrito no artigo (PREENCHER)
  # divergência com valores calculados por (PREENCHER) ...
  # ... se deve ao fato desta (PREENCHER)

  bsa = estimate_BSA(height, weight)
  eGFR = (ucr / scr) * (uvol / udur) * (1.73 / bsa)
  return eGFR

def estimate_ACR(ualb, udur, ucr, uvol):
  # parameters:
  #   ualb é uma medição do fluxo médio de albumina secretada por minuto durante a coleta de urina, em microgramas/min
  #   udur é a duração da coleta de urina, em minutos (valores típicos em torno de 660 a 720)
  #   ucr  é uma medição da concentração média de creatinina na amostra de urina coletada, em mg/dL
  #   uvol é o volume coletado de urina, em mL
  # Fórmula foi detalhada no relatório que o André enviou em 19/10/2021 e corrigiu em 13/3/2022.
  # divergência com valores calculados por https://www.mdapp.co/albumin-creatinine-ratio-calculator-461/ ...
  # ... se deve a arredondamentos
  eACR = 1E2 * (ualb * udur)/(ucr * uvol)
  return eACR

def estimate_BSA(height, weight):
  # parameters:
  #   height é a altura, em cm
  #   weight é o peso em Kg
  bsa = 0.007184 * weight ** 0.425 * height ** 0.725
  return bsa

def hasCKD(eGFR, eACR, hasDM, hasHTN, hadCS, hadCVA, hadMI):
  if(eGFR < 60.0 or eACR >= 30.0):
    # from the "prognosis of CKD" table in https://kdigo.org/wp-content/uploads/2017/02/KDIGO_2012_CKD_GL.pdf, pdf-page 9
    result = 2
  elif(eGFR < 90.0 and (hasDM + hasHTN + hadCS + hadCVA + hadMI > 0)):
    # this level is meant to represent individuals at some low CKD risk;
    # criterion from (PREENCHER!)
    result = 1
  else:
    result = 0
  return result

def load_PNS2013lab():

  def encodeGender(gender):
    # converts
    # 1 Masculino; 2 Feminino
    # to
    # 0 Masculino; 1 Feminino
    if(gender not in [1, 2]):
      return np.nan
    return 1 if gender == 2 else 0

  def encodeRace(race):
    # converts
    # 1 Branca; 2 Preta; 3 Amarela; 4 Parda; 5 Indígena
    # to
    # 0 Branco; 1 Nao-branco
    if(race not in [1, 2, 3, 4, 5]):
      return np.nan
    return 1 if race != 1 else 0

  # loads the dataset content from a tab-separated file
  # (content obtained from https://www.pns.icict.fiocruz.br/bases-de-dados/, click on "Exames" button)
  ds = pd.read_csv('../../datasets/PNS_2013.tsv', sep='\t', decimal='.')

  # discards columns that will not be used
  keepcolumns = ['Z001', 'Z002', 'Z003', 'Z004', 'Z005', 'Z025', 'Z048',
                 'Q030', 'Q002', 'Q063', 'Q068', 'Q066',
                 'F012', 'W00303', 'regiao', 'peso_lab',
                 ]
  ds.drop(ds.columns.difference(keepcolumns), axis=1, inplace=True)

  # renames columns to friendlier labels
  ds.rename(columns={'Z001':   'Gender',
                     'Z002':   'Age',
                     'Z003':   'Race',
                     'Z004':   'weight',
                     'Z005':   'height',
                     'Z025':   'SCr',
                     'Z048':   'UCr',
                     'Q030':   'hasDM',
                     'Q002':   'hasHTN',
                     'Q063':   'hadMI',
                     'Q068':   'hadCVA',
                     'Q066':   'hadCS',
                     'F012':   'BolsaFam',
                     'W00303': 'waist',
                    }, inplace=True)

  # recodes some columns to be compatible with the scheme in MDRD/CKD-EPI models,
  # (and, incidentally, identifies rows to be discarded because of invalid values)
  ds['Gender'] = ds.apply(lambda row: encodeGender(row.Gender), axis=1)
  ds['Race']   = ds.apply(lambda row: encodeRace(row.Race),     axis=1)
  ds['hasDM']  = ds.apply(lambda row: 1 if row.hasDM  == 1 else 0, axis=1)
  ds['hasHTN'] = ds.apply(lambda row: 1 if row.hasHTN == 1 else 0, axis=1)
  ds['hadMI']  = ds.apply(lambda row: 1 if row.hasDM  == 1 else 0, axis=1)
  ds['hadCVA'] = ds.apply(lambda row: 1 if row.hasHTN == 1 else 0, axis=1)
  ds['hadCS']  = ds.apply(lambda row: 1 if row.hasHTN == 1 else 0, axis=1)

  # identifies rows to be discarded (e.g., because of invalid values)
  for field in ['Age', 'BolsaFam', 'regiao']:
    ds[field] = pd.to_numeric(ds[field], errors='coerce', downcast = 'integer')
  for field in ['weight', 'height', 'SCr', 'UCr', 'waist', 'peso_lab']:
    ds[field] = pd.to_numeric(ds[field], errors='coerce', downcast = 'float')
  ds.dropna(inplace = True)

  # recasts some columns to other data types
  for field in ['regiao', 'Gender', 'Race', 'Age']:
    ds[field] = ds[field].astype(int)

  # extends the dataset with new derived columns
  ds['ID']     = ds.apply(lambda row: 'P{0:05d}'.format(row.name), axis=1)
  ds['eGFR']   = ds.apply(lambda row: estimate_GFR_CKDEPI(row.SCr, row.Age, row.Gender, row.Race), axis=1)
  ds['eBSA']   = ds.apply(lambda row: estimate_BSA(row.height, row.weight),  axis=1)
  ds['hasCKD'] = ds.apply(lambda row: hasCKD(row.eGFR, 0.0, row.hasDM, row.hasHTN, row.hadCS, row.hadCVA, row.hadMI), axis=1)

  # segments the dataset into groups on which the analysis is based
  ds['segment'] = ds.apply(lambda row: '{0}{1}'.format(int(row.Gender), int(row.Race)), axis=1)

  return ds

def load_ELSA():

  def encodeGender(gender):
    if(gender not in ['Feminino', 'Masculino']):
      return np.nan
    return 1 if gender == 'Feminino' else 0

  def encodeRace(race):
    if(race not in ['Branco', 'Amarelo', 'Indigena', 'Pardo', 'Preta']):
      return np.nan
    return 1 if race != 'Branco' else 0

  # loads the dataset content from a tab-separated file
  # (content is a sample of the ELSA Wave 2 dataset, obtained from researchers in the project)
  ds = pd.read_csv('../../datasets/DATASETELSA.tsv', sep='\t', decimal='.')

  # discards columns that will not be used
  keepcolumns = ['IDElsa', 'Idade', 'Sexo', 'Raca_Cor', 'Estatura', 'Peso',
                 'Creatinina_sangue', 'Cr_Ur', 'Microalbinuria_mcgmin', 'Vol_Ur', 'Tempo_Ur',
                 'Diabetes', 'Hipertensao', 'Infarto', 'AVC', 'Cirurgia Cardiaca', ]
  ds.drop(ds.columns.difference(keepcolumns), axis=1, inplace=True)

  # renames columns to friendlier labels
  ds.rename(columns={'IDElsa': 'ID',
                     'Idade': 'Age',
                     'Estatura': 'height',
                     'Peso': 'weight',
                     'Diabetes': 'hasDM',
                     'Hipertensao': 'hasHTN',
                     'Creatinina_sangue': 'SCr',
                     'Cr_Ur': 'UCr',
                     'Microalbinuria_mcgmin': 'UAlb',
                     'Vol_Ur': 'UVol',
                     'Tempo_Ur': 'UDur',
                     'Infarto': 'hadMI',
                     'AVC': 'hadCVA',
                     'Cirurgia Cardiaca': 'hadCS',
                    }, inplace=True)

  # recodes some columns to be compatible with the scheme in MDRD/CKD-EPI models,
  # (and, incidentally, identifies rows to be discarded because of invalid values)
  ds['Gender'] = ds.apply(lambda row: encodeGender(row.Sexo),   axis=1)
  ds['Race']   = ds.apply(lambda row: encodeRace(row.Raca_Cor), axis=1)
  ds.drop(columns=['Sexo','Raca_Cor'], inplace=True)
  ds['hasDM']  = ds.apply(lambda row: 1 if row.hasDM  == 'Sim' else 0, axis=1)
  ds['hasHTN'] = ds.apply(lambda row: 1 if row.hasHTN == 'Sim' else 0, axis=1)

  # identifies rows to be discarded (e.g., because of invalid values)
  for field in ['Age', 'hasDM', 'hasHTN', 'UCr', 'UVol', 'UDur', 'hadMI', 'hadCVA', 'hadCS']:
    ds[field] = pd.to_numeric(ds[field], errors='coerce', downcast = 'integer')
  for field in ['height', 'weight', 'SCr', 'UAlb']:
    ds[field] = pd.to_numeric(ds[field], errors='coerce', downcast = 'float')
  ds.dropna(inplace = True)

  # recasts some columns to appropriate data types
  for field in ['hadCS', 'hadCVA', 'hadMI', 'Gender', 'Race']:
    ds[field] = ds[field].astype(int)

  # extends the dataset with new derived columns
  ds['eGFR']   = ds.apply(lambda row: estimate_GFR_CKDEPI(row.SCr, row.Age, row.Gender, row.Race), axis=1)
  ds['eDCE']   = ds.apply(lambda row: estimate_GFR_DCE(row.UCr, row.SCr, row.UVol, row.UDur, row.height, row.weight), axis=1)
  ds['eACR']   = ds.apply(lambda row: estimate_ACR(row.UAlb, row.UDur, row.UCr, row.UVol),  axis=1)
  ds['eBSA']   = ds.apply(lambda row: estimate_BSA(row.height, row.weight),  axis=1)
  ds['hasCKD'] = ds.apply(lambda row: hasCKD(row.eGFR, row.eACR, row.hasDM, row.hasHTN, row.hadCS, row.hadCVA, row.hadMI), axis=1)

  # segments the dataset into groups on which the analysis is based
  ds['segment'] = ds.apply(lambda row: '{0}{1}'.format(row.Gender, row.Race), axis=1)
  #ds['segment'] = ds.apply(lambda row: '{0}{1}{2}'.format(row.Gender, row.Race, 1 if (row.hasDM + row.hasHTN + row.hadCS + row.hadCVA + row.hadMI) > 0 else 0), axis=1)

  return ds

def split(ds):

  healthy = ds.loc[(ds['hasCKD'] == 0)]['ID'].tolist()
  np.random.shuffle(healthy)
  num_of_ckdpos = len(ds.loc[(ds['hasCKD'] > 0)])
  healthy_te = healthy[:num_of_ckdpos]
  healthy_tr = healthy[num_of_ckdpos:]

  Tr_healthy   = ds.loc[(ds['ID'].isin(healthy_tr))]
  Te_healthy   = ds.loc[(ds['ID'].isin(healthy_te))]
  Te_unhealthy = ds.loc[(ds['hasCKD'] > 0)]

  return (Tr_healthy, Te_healthy, Te_unhealthy)

def doit(iter, Tr_healthy, Te_healthy, Te_unhealthy, params):

  # recover the model parameters
  (predVars, outcome, n_neighbors) = params

  def pm(agepairs):
    """
    Performance metric used in assessing the hypothesis
    """
    agediffs = [(age_real - age_pred) for (age_real, age_pred) in agepairs]    # bias
    #agediffs = [abs(age_real - age_pred) for (age_real, age_pred) in agepairs] # MAE
    return np.mean(agediffs)

    #agediffs = [(age_real - age_pred)**2 for (age_real, age_pred) in agepairs] # RMSE
    #return np.sqrt(np.mean(agediffs))

  def estimageAge(dists, attribs, neigh_attribs, neigh_ages):
    """
    Computes the average age of nearest neighbours weighted by their distances to the queried sample
    """
    epsilon = 1E-9
    dists_rec = 1/(dists[0] + epsilon)
    sum_dists_rec = dists_rec.sum()
    weights = dists_rec/sum_dists_rec
    eAge = neigh_ages.dot(weights)[0]

    #localModel = LinearRegression()
    #localModel.fit(neigh_attribs, neigh_ages.transpose(), weights)
    #eAge = localModel.predict(attribs)[0][0]

    #localModel = MLPRegressor(max_iter=100000, early_stopping = True)
    #localModel.fit(neigh_attribs, neigh_ages.ravel())
    #eAge = localModel.predict(attribs)[0]

    return eAge

  pm_neg = {}
  pm_pos = {}
  partsizes = defaultdict(dict)
  segments = Tr_healthy['segment'].unique()
  preds = []
  buffer = '{0}\t#{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}'
  for segment in sorted(segments):

    # trains a nearest neighbour model using data from a single segment of Tr_healthy
    model = NearestNeighbors(n_neighbors = n_neighbors, algorithm = 'auto', metric = 'minkowski', p = 2)
    data  = Tr_healthy.loc[Tr_healthy.segment == segment]
    tr_ids = data['ID'].to_numpy()
    X = data[predVars].to_numpy()
    y = data[outcome ].to_numpy()
    scaler = StandardScaler(with_mean = False, with_std = False).fit(X)
    X = scaler.transform(X)
    model.fit(X)

    # determines the size of the test samples for the current segment
    n_ckdneg = len(Te_healthy.loc[    Te_healthy.segment == segment])
    n_ckdpos = len(Te_unhealthy.loc[Te_unhealthy.segment == segment])
    ss = min(n_ckdneg, n_ckdpos)

    # evaluates the performance metric on a sample of healthy individuals (CKD-)
    data = Te_healthy.loc[Te_healthy.segment == segment]
    te_ids = data['ID'].to_numpy()
    Xneg = data[predVars].to_numpy()
    yneg = data[outcome ].to_numpy()
    Xneg = scaler.transform(Xneg)
    partsizes['ckd-'][segment] = ss
    agepairs = []
    for i in range(ss):
      #print(i)
      dists, neighbors = model.kneighbors(Xneg[i].reshape(1, -1), return_distance=True)
      eAge = estimageAge(dists, Xneg[i].reshape(1, -1), X[neighbors[0]], y[neighbors])
      agepairs.append((yneg[i], eAge))
      preds.append(buffer.format(iter, segment, 'CKD-', te_ids[i], yneg[i], Xneg[i], eAge, tr_ids[neighbors[0]], dists[0], y[neighbors[0]], np.array2string(X[neighbors[0]]).replace('\n', ' '), y[neighbors[0]].mean()))
    pm_neg[segment] = pm(agepairs)

    # evaluates the performance metric on a sample of unhealthy individuals (CKD+)
    data = Te_unhealthy.loc[Te_unhealthy.segment == segment]
    te_ids = data['ID'].to_numpy()
    Xpos = data[predVars].to_numpy()
    ypos = data[outcome ].to_numpy()
    Xpos = scaler.transform(Xpos)
    partsizes['ckd+'][segment] = ss
    agepairs = []
    for i in range(ss):
      #print(i)
      dists, neighbors = model.kneighbors(Xpos[i].reshape(1, -1), return_distance=True)
      eAge = estimageAge(dists, Xpos[i].reshape(1, -1), X[neighbors[0]], y[neighbors])
      agepairs.append((ypos[i], eAge))
      preds.append(buffer.format(iter, segment, 'CKD+', te_ids[i], ypos[i], Xpos[i], eAge, tr_ids[neighbors[0]], dists[0], y[neighbors[0]], np.array2string(X[neighbors[0]]).replace('\n', ' '), y[neighbors[0]].mean()))
    pm_pos[segment] = pm(agepairs)

  return (pm_neg, pm_pos, partsizes, preds)

def assess_hyphothesis(pm_neg_segs, pm_pos_segs, tries, partsizes):

  pm_neg_values = defaultdict(list)
  pm_pos_values = defaultdict(list)
  segments = []
  for i in range(tries):
    for segment in pm_neg_segs[i]:
      segments.append(segment)
      pm_neg_values[segment].append(pm_neg_segs[i][segment])
      pm_pos_values[segment].append(pm_pos_segs[i][segment])

  segments = sorted(set(segments))
  responses = []
  cis_pm_neg = []
  cis_pm_pos = []
  for segment in segments:
    ci_pm_neg = bootstrap([pm_neg_values[segment],], np.mean).confidence_interval
    ci_pm_pos = bootstrap([pm_pos_values[segment],], np.mean).confidence_interval
    hypothesis = ci_pm_neg.high < ci_pm_pos.low
    complement = 'CKD- [{0:4.1f}, {1:4.1f}] #{2:4d}, CKD+ [{3:4.1f}, {4:4.1f}] #{5:4d}'.format(
                       ci_pm_neg.low, ci_pm_neg.high, partsizes['ckd-'][segment],
                       ci_pm_pos.low, ci_pm_pos.high, partsizes['ckd+'][segment])
    if(hypothesis):
      res = '*** Segment {0}: Yes, the hypothesis is supported by the results: {1}'.format(segment, complement)

    else:
      res = '*** Segment {0}: No, the hypothesis is NOT supported by the results: {1}'.format(segment, complement)
    responses.append(res)
    cis_pm_neg.append(ci_pm_neg)
    cis_pm_pos.append(ci_pm_pos)

  return responses, cis_pm_neg, cis_pm_pos

def main(dataset, n_neighbors, tries):

  ECO_SEED = 20
  np.random.seed(ECO_SEED)
  np.set_printoptions(precision=5, suppress=True, linewidth=3000)

  print()
  if dataset.lower() == 'elsa':
    print('Loading and preprocessing the ELSA dataset')
    ds = load_ELSA()
  elif(dataset.lower() == 'pns'):
    print('Loading and preprocessing the PNS 2013 lab exams dataset')
    ds = load_PNS2013lab()
  print(ds.info())

  serialise(ds, 'dataset')

  num_of_ckdpos = len(ds.loc[(ds['hasCKD'] >  0)])
  num_of_ckdneg = len(ds.loc[(ds['hasCKD'] == 0)])
  print('-- number of CKD+ individuals: {0:5d}'.format(num_of_ckdpos))
  print('-- number of CKD- individuals: {0:5d}'.format(num_of_ckdneg))

  print('Learning and assessing a predictive model for kidney ageing')

  #predVars = ['height', 'weight']
  #outcome  = 'eBSA'

  #predVars = ['eBSA']
  #outcome  = 'weight'
  #outcome  = 'height'

  #predVars = ['height', 'weight', 'SCr', 'UCr', 'UVol', 'UDur']
  #outcome  = 'eDCE'

  # xxx
  # por que (height,weight) prediz eBSA com precisão, e
  # (height,weight,SCr,UCr,UVol,UDur) não prediz eDCE com a mesma precisão?

  #predVars = ['height', 'weight', 'SCr']
  #outcome  = 'Age'

  predVars = ['eBSA', 'SCr', 'UCr']
  outcome  = 'Age'

  params = (predVars, outcome, n_neighbors)
  pm_neg_segs = []
  pm_pos_segs = []
  buffer = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\t{10}\t{11}'
  header = buffer.format('iter', 'segment', 'group', 'id', 'age', 'attrs', '*age', 'neighbors-ids', 'neighbors-dists', 'neighbors-ages', 'neighbors-attrs', 'neighbors-ages-mean')
  predictions = [header]
  for iter in range(tries):
    print('-- iteration {0:2d}'.format(iter+1))
    (Tr_healthy, Te_healthy, Te_unhealthy) = split(ds)
    (pm_neg, pm_pos, partsizes, preds) = doit(iter+1, Tr_healthy, Te_healthy, Te_unhealthy, params)
    pm_neg_segs.append(pm_neg)
    pm_pos_segs.append(pm_pos)
    predictions = predictions + preds
  saveAsText('\n'.join(predictions), 'trace.csv')

  # checks if the hypothesis is supported by the obtained results
  print('Assessing the hypothesis')
  (responses, _, _) = assess_hyphothesis(pm_neg_segs, pm_pos_segs, tries, partsizes)
  for res in responses:
    print(res)

if(__name__ == '__main__'):

  dataset = sys.argv[1]
  n_neighbors = int(sys.argv[2])
  tries = int(sys.argv[3])
  main(dataset, n_neighbors, tries)
