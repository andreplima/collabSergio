import re
import pickle
import codecs
import numpy  as np
import pandas as pd
import statsmodels.api as sm

from copy        import copy
from os.path     import join
from collections import defaultdict
from scipy.stats import kendalltau
from matplotlib  import pyplot as plt
from sklearn.metrics        import mean_squared_error, r2_score
from sklearn.decomposition  import PCA
from matplotlib.collections import LineCollection

ECO_SEED = 23
ECO_FONT_SIZE = 18

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
  # estimator described in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2763564/, page 14 (table footnote)
  # divergence bwith values obtained from https://www.sbn.org.br/profissional/utilidades/calculadoras-nefrologicas/ ...
  # ... are owing to the fact of the latter using the 8 specific equations (in which the constant is rounded to integer)

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

def estimate_BSA(weight, height):
  # parameters:
  #   weight in kG
  #   height in cm
  # returns estimate of body surface area, in m^2
  return 0.007184 * weight ** 0.425 * height ** 0.725

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

  eBSA = estimate_BSA(weight, height)
  eGFR = (ucr /scr) * (uvol / udur) * (1.73 / eBSA)
  return eGFR

def estimate_ACR(ualb, udur, ucr, uvol):
  # parameters:
  #   ualb é uma medição do fluxo médio de albumina secretada por minuto durante a coleta de urina, em microgramas/minuto
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
    result = 1
  elif(eGFR < 90.0 and (hasDM + hasHTN + hadCS + hadCVA + hadMI > 0)):
    # this group is meant to represent individuals at some low CKD risk; criterion from (PREENCHER!)
    result = 2
  else:
    result = 0
  return result

def readELSA(targetpath, filename):

  # loads the dataset
  df = pd.read_csv(join(*targetpath, filename), sep='\t', decimal='.')
  #ds.info()

  # remove columns that will not be used
  df = df.drop(columns=['Codigo Sexo', 'Microalbinuria_mcgmin',
                        'Febre reumática', 'Creatinina por kg/dia'])

  # rename some column names as abbreviated labels
  # (abbreviations based on https://medlineplus.gov/appendixb.html)
  df = df.rename(columns={'Creatinina_sangue': 'SCr',
                          'Microalbinuria_mcgmin.1': 'UAlb',
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
                          'Peso': 'weight',
                          'Sodio_sangue': 'SNa',
                          'Sodio_Ur': 'UNa',
                          'K_Ur': 'UK',
                          'Colesterol': 'TChol',
                          'HDL': 'HDL',
                          'LDL': 'LDL',
                          'TG': 'TG',
                          'TG_HDL': 'TG_HDL',
                          'Sódio_24h': 'Na24h',
                          'Potassio_24h': 'K24h',
                          'Acido_urico': 'UH+',
                         })

  # converts some columns to numerical values (or np.nan otherwise)
  for field in ['UAlb', 'hadCS', 'hadCVA', 'hadMI', 'height', 'weight']:
    df[field] = pd.to_numeric(df[field], errors='coerce')

  df['hasDM']  = df.apply(lambda row: 0 if row.hasDM  == 'Nao' else 1, axis=1)
  df['hasHTN'] = df.apply(lambda row: 0 if row.hasHTN == 'Nao' else 1, axis=1)
  df['Gender'] = df.apply(lambda row: encodeGender(row.Sexo),   axis=1)
  df['Race']   = df.apply(lambda row: encodeRace(row.Raca_Cor), axis=1)
  df = df.drop(columns=['Sexo','Raca_Cor'])

  # discard rows with np.nan values in some column
  df = df.dropna()

  # converts some columns to integer values
  for field in ['hasDM', 'hasHTN', 'hadCS', 'hadCVA', 'hadMI', 'Gender', 'Race']:
    df[field] = df[field].astype(int)

  # extends the dataset to include computed columns
  df['eGFR'] = df.apply(lambda row: estimate_GFR_CKDEPI(row.SCr, row.Age, row.Gender, row.Race), axis=1)
  df['eDCE'] = df.apply(lambda row: estimate_GFR_DCE(row.UCr, row.SCr, row.UVol, row.UDur, row.height, row.weight), axis=1)
  df['eACR'] = df.apply(lambda row: estimate_ACR(row.UAlb, row.UDur, row.UCr, row.UVol),  axis=1)
  df['eBSA'] = df.apply(lambda row: estimate_BSA(row.weight, row.height),  axis=1)
  df['hasCKD'] = df.apply(lambda row: hasCKD(row.eGFR, row.eACR, row.hasDM, row.hasHTN, row.hadCS, row.hadCVA, row.hadMI), axis=1)

  df['regBA'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'BA' else 0, axis=1)
  df['regES'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'ES' else 0, axis=1)
  df['regMG'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'MG' else 0, axis=1)
  df['regRJ'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'RJ' else 0, axis=1)
  df['regRS'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'RS' else 0, axis=1)
  df['regSP'] = df.apply(lambda row: 1 if row.IDElsa[-2:] == 'SP' else 0, axis=1)

  return df

def getDomains(vars, df):
  return {var: (df[var].min(), df[var].max()) for var in vars}

def computeMetrics(Yreal, Ypred):
  print('\nAvaliação do modelo com dados de teste:')
  print('** MSE = {0:5.1f}'.format(mean_squared_error(Yreal, Ypred)))
  print('** R-squared: {0:5.3f}'.format(r2_score(Yreal, Ypred)))
  return None

def predByAge(Yreal, Ypred):
  age_pairs = [(age_real, age_pred) for (age_real, age_pred) in sorted(zip(Yreal.to_list(), Ypred.to_list()))]
  age_slots = defaultdict(list)

  ses = []
  for (age_real, age_pred) in age_pairs:
    se = (age_real - age_pred)**2
    ses.append(se)
    #print(age_real, int(age_pred), se)
  print(np.mean(ses))

  for (age_real, age_pred) in age_pairs:
    age_slots[age_real].append(int(age_pred))

  for age_real in age_slots:
    age_slots[age_real] = sorted(set(age_slots[age_real]))
    print('{0}: {1}'.format(age_real, age_slots[age_real]))

  return None

def plotPanel(configs, plotTitle, rowLbls = None, domains = None, legend = True):

  npans = len(configs)
  nrows = len(rowLbls)
  ncols = len(configs) // nrows
  fig, axes = plt.subplots(nrows, ncols, figsize=(24, 8), sharey=True)
  innerseps = {'left': 0.03, 'bottom': 0.12, 'right': 0.98, 'top': 0.92, 'wspace': 0.10, 'hspace': 0.15}
  plt.subplots_adjust(left   = innerseps['left'],
                      bottom = innerseps['bottom'],
                      right  = innerseps['right'],
                      top    = innerseps['top'],
                      wspace = innerseps['wspace'],
                      hspace = innerseps['hspace'])

  fig.suptitle(plotTitle, fontsize=ECO_FONT_SIZE)

  for i in range(npans):
    pos = i + 1
    plt.subplot(nrows, ncols, pos)
    (xlabel, ylabel, xs, ys, eys) = configs[i]
    if(legend):
      plt.gca().scatter(xs, eys, color='r', marker='.', label='Estimated', alpha=0.2)
      plt.gca().scatter(xs,  ys, color='b', marker='.', label='Measured')
    else:
      plt.gca().scatter(xs, eys, color='r', marker='.', alpha=0.2)
      plt.gca().scatter(xs,  ys, color='b', marker='.')

    if(i >= npans - ncols):
      plt.gca().set_xlabel(xlabel, fontsize=ECO_FONT_SIZE)
      #xxx set limits

    if(i % ncols == 0):
      rowLabel = rowLbls[i // ncols]
      plt.gca().set_ylabel(ylabel + ' ' + rowLabel, fontsize=ECO_FONT_SIZE)

    try:
      var = xlabel
      (lb, ub) = domains[var]
      plt.xlim(lb, ub)
    except KeyError:
      None

    plt.gca().grid(True)
    if(legend):
      plt.gca().legend()

    errors = []
    for j in range(len(xs)):
      measured  = (xs.values[j],   ys.values[j])
      predicted = (xs.values[j],  eys.values[j])
      errors.append([measured, predicted])
    glc = LineCollection(errors, colors = ['red' for _ in errors], linewidth = .8, alpha=0.2)
    plt.gca().add_collection(glc)
    #plt.gca().legend()

  plt.show()

  return None

def plotPanelPerf(outcome, predVars, test, Ypred, valid, Vpred, plotTitle, domains = None, legend = True):

  configs = []
  rowLbls = []

  rowLbls.append(' (DRC-)')
  for label in [outcome] + predVars:
    configs.append((label, outcome, test[label], test[outcome], Ypred))

  rowLbls.append(' (DRC+)')
  for label in [outcome] + predVars:
    configs.append((label, outcome, valid[label], valid[outcome], Vpred))

  plotPanel(configs, plotTitle, rowLbls, domains, legend)
  #predByAge(Yreal, Ypred)

  return None

def LRModel(outcome, predVars, train, test, valid, plotTitle, domains = None):

  Y = train[outcome]
  X = train[predVars]
  model = sm.GLS(Y, sm.add_constant(X))
  results = model.fit()
  print(results.summary())

  Yreal = test[outcome]
  Ypred = results.predict(sm.add_constant(test[predVars]))
  computeMetrics(Yreal, Ypred)

  Vreal = valid[outcome]
  Vpred = results.predict(sm.add_constant(valid[predVars]))
  computeMetrics(Vreal, Vpred)

  plotPanelPerf(outcome, predVars, test, Ypred, valid, Vpred, plotTitle, domains)

  return results

# returns Xnew,Ynew arrays resulted from applying SRT-transformations on df

ECO_FENCE = 0.9

def SRT(df, predVars, outcome, activation = 'identity', ndims = None, params = None):

  X = df[predVars].to_numpy(copy=True)
  Y = df[outcome].to_numpy(copy=True)

  if(ndims == None):
    ndims = len(predVars)

  if(params == None):
    # determines the parameters of the transformations
    mu   = np.mean(X, axis=0)
    pca  = PCA(n_components = ndims, whiten = True)
    (lb, ub) = (Y.min(), Y.max())
  else:
    (mu, pca, activation, lb, ub) = params

  # applies SRT operations to predictors
  Xnew = pca.fit_transform(X - mu)

  # applies a scaling operation to the outcome
  # see some function definitions in https://openstax.org/books/calculus-volume-1/pages/1-5-exponential-and-logarithmic-functions
  if(activation == 'identity'):
    Ynew = Y
  elif(activation == 'logistic'):
    Ynew = (2 * ECO_FENCE - 1) * (Y - lb)/(ub - lb) + (1 - ECO_FENCE)
  elif(activation == 'tanh'):
    Ynew = (2 * ECO_FENCE) * (Y - lb)/(ub - lb) - 1
  elif(activation == 'relu'):
    Ynew = Y - lb

  # prepares the outputs
  params = (mu, pca, activation, lb, ub)
  return (Xnew, Ynew, params)

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

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res
