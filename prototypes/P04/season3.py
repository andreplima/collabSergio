import sys
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors

#import statsmodels.api as sm

#from copy import copy
#from collections import defaultdict
#from matplotlib import pyplot as plt
#from matplotlib.collections import LineCollection
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
#from sklearn.decomposition import PCA
#from sklearn.neural_network import MLPRegressor

def encodeGender(gender):
  if(gender not in ['Feminino', 'Masculino']):
    return np.nan
  return 1 if gender == 'Feminino' else 0

def encodeRace(race):
  if(race not in ['Branco', 'Amarelo', 'Indigena', 'Pardo', 'Preta']):
    return np.nan
  return 1 if race != 'Branco' else 0

def estimate_GFR_CKDEPI(scr, age, gender, race):
  # parameters:
  #   scr is serum creatinine measured in mg/dL using (PREENCHER tipo de assay, see https://youtu.be/KcoeVtiYGpU)
  #   age in years, gender and race as strings in the domains described above
  #   gender: female is encoded as 1, and male as 0
  #   race: white is encoded as 0, and non-whites (including blacks) as 1
  #   -- returns estimated GFR in mL/min/1.73m^2
  # estimador descrito no artigo https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763564/, página 14 (rodapé da tabela)
  # divergência com valores calculados por https://www.sbn.org.br/profissional/utilidades/calculadoras-nefrologicas/ ...
  # ... se deve ao fato desta usar as 8 equações individuais, nas quais o primeiro coeficiente é arredondado

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

# Calcula a idade a partir das demais variáveis da equação CKD-EPI, por manipulação algébrica
def estimate_Age_CKDEPI(eGFR, scr, gender, race):
  kappa = lambda gender:  0.7   if gender == 1 else  0.9
  alpha = lambda gender: -0.329 if gender == 1 else -0.411
  scr_k = scr/kappa(gender)
  age = (1 / np.log(0.993) *
          np.log(eGFR / (141
                        * min(scr_k, 1) ** alpha(gender)
                        * max(scr_k, 1) ** -1.209
                        * (1.018 if gender == 1 else 1.0)
                        * (1.159 if race   == 1 else 1.0)
                        )
                )
        )
  return round(age, 0)

def estimate_BSA(height, weight):
  # parameters:
  #   height é a altura, em cm
  #   weight é o peso em kG
  bsa = 0.007184 * weight ** 0.425 * height ** 0.725
  return bsa

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

def hasCKD(eGFR, eACR, hasDM, hasHTN, hadCS, hadCVA, hadMI):
  if(eGFR < 60.0 or eACR >= 30.0):
    # from the "prognosis of CKD" table in https://kdigo.org/wp-content/uploads/2017/02/KDIGO_2012_CKD_GL.pdf, pdf-page 9
    result = 2
  elif(eGFR < 90.0 and (hasDM + hasHTN + hadCS + hadCVA + hadMI > 0)):
    # this group is meant to represent individuals at some low CKD risk;
    # criterion from (PREENCHER!)
    result = 1
  else:
    result = 0
  return result

#
def load_ELSA():

  ds = pd.read_csv('../../datasets/DATASETELSA.tsv', sep='\t', decimal='.')

  #
  ds.drop(columns=['Codigo Sexo',
                   'Acido_urico', 'Colesterol', 'TG', 'HDL', 'LDL', 'TG_HDL',
                   'Sodio_sangue', 'Sodio_Ur', 'K_Ur','Sódio_24h',
                   'Potassio_24h', 'Febre reumática'], inplace = True)

  #
  ds.rename(columns={'Creatinina_sangue': 'SCr',
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

  return ds

def split(ds):
  num_of_ckdpos = len(ds.loc[(ds['hasCKD'] >  0)])

  healthy = ds.loc[(ds['hasCKD'] == 0)]['IDElsa'].tolist()
  healthy_te = np.random.choice(healthy, size=num_of_ckdpos, replace = False)
  healthy_tr = [id for id in healthy if id not in healthy_te]

  Tr_healthy   = ds.loc[(ds['IDElsa'].isin(healthy_tr))]
  Te_healthy   = ds.loc[(ds['IDElsa'].isin(healthy_te))]
  Te_unhealthy = ds.loc[(ds['hasCKD'] >  0)]

  return (Tr_healthy, Te_healthy, Te_unhealthy)

def doit(Tr_healthy, Te_healthy, Te_unhealthy, params):

  (predVars, outcome, nn) = params

  model = NearestNeighbors(n_neighbours = nn)
  X = Tr_healthy[predVars]
  y = Tr_healthy[outcome]
  model.fit(X, y)

  v1 = 1
  v2 = 2
  return (v1, v2)

def hyphothesis(v1_values, v2_values):
  return True

def main(tries):

  print('Loading and preprocessing the ELSA dataset.')
  dataset = load_ELSA()
  #print(dataset.info())

  print('Splitting the dataset into training and test partitions')
  (Tr_healthy, Te_healthy, Te_unhealthy) = split(dataset)
  print('-- number of CKD+ individuals: {0}'.format(len(Te_unhealthy)))
  print('-- number of CKD- individuals: {0}'.format(len(Tr_healthy)+len(Te_healthy)))
  print('-- Test partition comprises {0} healthy and {1} unhealthy individuals'.format(len(Te_healthy), len(Te_unhealthy)))

  #
  params = (['eDCE', 'SCr', 'Gender', 'Race'], 'Age', 3)
  v1_values = []
  v2_values = []
  for i in range(tries):
    (v1, v2) = doit(Tr_healthy, Te_healthy, Te_unhealthy, params)
    v1_values.append(v1)
    v2_values.append(v2)

  # checks if the hypothesis is supported by the obtained results
  if(hyphothesis(v1_values, v2_values)):
    print('*** Yes, the hypothesis has been supported by the obtained results')
  else:
    print('*** No, the hypothesis has NOT been supported by the obtained results')

if(__name__ == '__main__'):

  tries = int(sys.argv[1])
  main(tries)
