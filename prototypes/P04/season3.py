# next task - make loaders as similar as possible
import sys
import numpy as np
import pandas as pd

from os.path     import exists
from collections import defaultdict
from scipy.stats import bootstrap
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings('error')

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
  #   ucr is unine creatinine measured in mg/dL using (PREENCHER tipo de assay, see https://youtu.be/KcoeVtiYGpU)
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
  eGFR = (ucr /scr) * (uvol / udur) * (1.73 / bsa)
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
  # https://www.pns.icict.fiocruz.br/bases-de-dados/

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

  #
  ds = pd.read_csv('../../datasets/PNS_2013.tsv', sep='\t', decimal='.')

  keepcolumns = ['Z001', 'Z002', 'Z003', 'Z004', 'Z005', 'Z025', 'Z048',
                 'F012', 'W00303',
                 'regiao', 'peso_lab']
  ds.drop(ds.columns.difference(keepcolumns), axis=1, inplace=True)

  # ajusta valores de algumas colunas (converte para número ou nan)
  for field in keepcolumns:
    ds[field] = pd.to_numeric(ds[field], errors='coerce')

  # remove linhas com dados inválidos (descarte)
  ds.dropna(inplace = True)

  #
  ds.rename(columns={'Z001':   'Gender',
                     'Z002':   'Age',
                     'Z003':   'Race',
                     'Z004':   'weight',
                     'Z005':   'height',
                     'Z025':   'SCr',
                     'Z048':   'UCr',
                     'F012':   'BolsaFam',
                     'W00303': 'waist',
                    }, inplace=True)

  # ajusta valores de algumas colunas (troca tipo de dados)
  for field in ['regiao']:
    ds[field] = ds[field].astype(int)
  ds['Gender'] = ds.apply(lambda row: encodeGender(row.Gender), axis=1)
  ds['Race']   = ds.apply(lambda row: encodeRace(row.Race),     axis=1)

  # estende o dataset para incluir novas colunas
  ds['ID'] = ds.index
  ds['eGFR']   = ds.apply(lambda row: estimate_GFR_CKDEPI(row.SCr, row.Age, row.Gender, row.Race), axis=1)
  ds['eBSA']   = ds.apply(lambda row: estimate_BSA(row.height, row.weight),  axis=1)
  #ds['hasCKD'] = ds.apply(lambda row: hasCKD(row.eGFR, 0.0, row.hasDM, row.hasHTN, row.hadCS, row.hadCVA, row.hadMI), axis=1)
  ds['hasCKD'] = ds.apply(lambda row: hasCKD(row.eGFR, 0.0, 0, 0, 0, 0, 0), axis=1)

  # segments the dataset in gender and race groups
  ds['segment'] = ds.apply(lambda row: '{0}{1}'.format(row.Gender, row.Race), axis=1)
  #ds['segment'] = ds.apply(lambda row: '{0}{1}{2}'.format(row.Gender, row.Race, 1 if (row.hasDM + row.hasHTN + row.hadCS + row.hadCVA + row.hadMI) > 0 else 0), axis=1)

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

  #
  ds = pd.read_csv('../../datasets/DATASETELSA.tsv', sep='\t', decimal='.')

  #
  ds.drop(columns=['Codigo Sexo',
                   'Acido_urico', 'Colesterol', 'TG', 'HDL', 'LDL', 'TG_HDL',
                   'Sodio_sangue', 'Sodio_Ur', 'K_Ur','Sódio_24h',
                   'Potassio_24h', 'Febre reumática', 'Microalbinuria_mcgmin.1',
                   'Creatinina por kg/dia'], inplace = True)

  #
  ds.rename(columns={'IDElsa': 'ID',
                     'Creatinina_sangue': 'SCr',
                     'Microalbinuria_mcgmin': 'UAlb',
                     'Cr_Ur': 'UCr',
                     'Tempo_Ur': 'UDur',
                     'Vol_Ur': 'UVol',
                     'Idade': 'Age',
                     'Diabetes': 'hasDM',
                     'Hipertensao': 'hasHTN',
                     'Cirurgia Cardiaca': 'hadCS',
                     'AVC': 'hadCVA',
                     'Infarto': 'hadMI',
                     'Estatura': 'height',
                     'Peso': 'weight'}, inplace=True)

  # ajusta valores de algumas colunas (converte para número ou nan)
  for field in ['UAlb', 'hadCS', 'hadCVA', 'hadMI', 'height', 'weight']:
    ds[field] = pd.to_numeric(ds[field], errors='coerce')

  ds['hasDM']  = ds.apply(lambda row: 0 if row.hasDM  == 'Nao' else 1, axis=1)
  ds['hasHTN'] = ds.apply(lambda row: 0 if row.hasHTN == 'Nao' else 1, axis=1)
  ds['Gender'] = ds.apply(lambda row: encodeGender(row.Sexo),   axis=1)
  ds['Race']   = ds.apply(lambda row: encodeRace(row.Raca_Cor), axis=1)
  ds.drop(columns=['Sexo','Raca_Cor'], inplace = True)

  # remove linhas com dados inválidos (descarte)
  ds.dropna(inplace = True)

  # ajusta valores de algumas colunas (troca tipo de dados)
  for field in ['hasDM', 'hasHTN', 'hadCS', 'hadCVA', 'hadMI', 'Gender', 'Race']:
    ds[field] = ds[field].astype(int)

  # estende o dataset para incluir novas colunas
  ds['eGFR']   = ds.apply(lambda row: estimate_GFR_CKDEPI(row.SCr, row.Age, row.Gender, row.Race), axis=1)
  ds['eDCE']   = ds.apply(lambda row: estimate_GFR_DCE(row.UCr, row.SCr, row.UVol, row.UDur, row.height, row.weight), axis=1)
  ds['eACR']   = ds.apply(lambda row: estimate_ACR(row.UAlb, row.UDur, row.UCr, row.UVol),  axis=1)
  ds['eBSA']   = ds.apply(lambda row: estimate_BSA(row.height, row.weight),  axis=1)
  ds['hasCKD'] = ds.apply(lambda row: hasCKD(row.eGFR, row.eACR, row.hasDM, row.hasHTN, row.hadCS, row.hadCVA, row.hadMI), axis=1)

  # segments the dataset in gender and race groups
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

def doit(Tr_healthy, Te_healthy, Te_unhealthy, params):

  # recover the model parameters
  (predVars, outcome, n_neighbors) = params

  def pm(agepairs):
    """
    Performance metric used in assessing the hypothesis
    """
    #print(agepairs[0:3])
    agediffs = [(age_real - age_pred) for (age_real, age_pred) in agepairs]    # bias
    #agediffs = [abs(age_real - age_pred) for (age_real, age_pred) in agepairs] # MAE
    return np.mean(agediffs)

    #agediffs = [(age_real - age_pred)**2 for (age_real, age_pred) in agepairs] # RMSE
    #return np.sqrt(np.mean(agediffs))

  def estimageAge(dists, ages):
    """
    Computes the average age of nearest neighbours weighted by their distances to the queried sample
    """
    epsilon = 1E-9
    dists_rec = 1/(dists[0] + epsilon)
    sum_dists_rec = dists_rec.sum()
    weights = dists_rec/sum_dists_rec
    eAge = ages.dot(weights)[0]
    return eAge

  pm_neg = {}
  pm_pos = {}
  partsizes = defaultdict(dict)
  segments = Tr_healthy['segment'].unique()
  for segment in sorted(segments):

    # trains a nearest neighbour model using data from Tr_healthy (only healthy individuals)
    model = NearestNeighbors(n_neighbors = n_neighbors, algorithm = 'auto', metric = 'minkowski', p = 2)
    X = Tr_healthy[predVars].loc[(Tr_healthy.segment == segment)].to_numpy()
    y = Tr_healthy[outcome ].loc[(Tr_healthy.segment == segment)].to_numpy()
    model.fit(X)

    # evaluates the performance metric of age prediction for healthy individuals
    Xneg = Te_healthy[predVars].loc[(Te_healthy.segment == segment)].to_numpy()
    yneg = Te_healthy[outcome ].loc[(Te_healthy.segment == segment)].to_numpy()
    (nrows, _) = Xneg.shape
    partsizes['ckd-'][segment] = nrows
    agepairs = []
    for i in range(nrows):
      dists, neighbors = model.kneighbors(Xneg[i].reshape(1, -1), return_distance=True)
      eAge = estimageAge(dists, y[neighbors])
      agepairs.append((yneg[i], eAge))
    pm_neg[segment] = pm(agepairs)

    # evaluates the performance metric of age prediction for unhealthy individuals
    Xpos = Te_unhealthy[predVars].loc[(Te_unhealthy.segment == segment)].to_numpy()
    ypos = Te_unhealthy[outcome ].loc[(Te_unhealthy.segment == segment)].to_numpy()
    (nrows, _) = Xpos.shape
    partsizes['ckd+'][segment] = nrows
    agepairs = []
    for i in range(nrows):
      dists, neighbors = model.kneighbors(Xpos[i].reshape(1, -1), return_distance=True)
      eAge = estimageAge(dists, y[neighbors])
      agepairs.append((ypos[i], eAge))
    pm_pos[segment] = pm(agepairs)

  return (pm_neg, pm_pos, partsizes)

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
    try:
      ci_pm_neg = bootstrap([pm_neg_values[segment],], np.mean).confidence_interval
      ci_pm_pos = bootstrap([pm_pos_values[segment],], np.mean).confidence_interval
    except RuntimeWarning:
      breakpoint()
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

  if dataset.lower() == 'elsa':
    print('Loading and preprocessing the PNS 2013 lab exams dataset')
    ds = load_ELSA()
  elif(dataset.lower() == 'pns'):
    print('Loading and preprocessing the ELSA dataset')
    ds = load_PNS2013lab()
  print(ds.info())

  num_of_ckdpos = len(ds.loc[(ds['hasCKD'] >  0)])
  num_of_ckdneg = len(ds.loc[(ds['hasCKD'] == 0)])
  print('-- number of CKD+ individuals: {0:5d}'.format(num_of_ckdpos))
  print('-- number of CKD- individuals: {0:5d}'.format(num_of_ckdneg))

  print('Learning and assessing the model of renal age')
  #predVars = ['eDCE', 'SCr', 'eACR', 'height', 'weight']
  predVars = ['height', 'weight', 'SCr']
  outcome  = 'Age'
  params = (predVars, outcome, n_neighbors)
  pm_neg_segs = []
  pm_pos_segs = []
  for i in range(tries):
    print('-- iteration {0:2d}'.format(i+1))
    (Tr_healthy, Te_healthy, Te_unhealthy) = split(ds)
    (pm_neg, pm_pos, partsizes) = doit(Tr_healthy, Te_healthy, Te_unhealthy, params)
    pm_neg_segs.append(pm_neg)
    pm_pos_segs.append(pm_pos)

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
