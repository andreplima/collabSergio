"""
  This script is based on epidemiological parameters published on the literature on biomedical sciences.
  Particularly, it assumes that data from the following articles are valid for the Brazilian population:

  [1] Sandhi M Barreto, Roberto M Ladeira, Bruce B Duncan, Maria Ines Sch-midt, Antonio A Lopes,
      Isabela M Benseñor, Dora Chor, Rosane H Griep,Pedro G Vidigal, Antonio L Ribeiro, Paulo A Lotufo, and
      José Geraldo Mill. Chronic kidney disease among adult participants of the ELSA-brasil cohort:
      association with race and socioeconomic position. Journal of Epidemiology & Community Health,
      70(4):380–389, 2016.

  [2] Andrew S Levey, Lesley A Stevens, Christopher H Schmid, Yaping Zhang, Alejandro F Castro III,
      Harold I Feldman, John W Kusek, Paul Eggers, Frederick Van Lente, Tom Greene, et al.
      A new equation to estimate glomerular filtration rate. Annals of Internal Medicine,
      150(9):604–612, 2009

  [3] Efron, Bradley, and Robert J. Tibshirani. An introduction to the Bootstrap. Chapman & Hall/CRC (1993)

"""
import numpy  as np
import pandas as pd

from os.path     import join
from collections import namedtuple
from scipy.stats import norm, bootstrap, t as tdist

ECO_SEED   = 31
ECO_FEMALE = 'Feminino'
ECO_BLACK  = 'Preta'

ECO_RECORD_FIELDS = ['eGFR', 'ACR', 'DRC_C1', 'DRC_C2', 'DRC_AND', 'DRC_OR']
Record            = namedtuple('Record', ECO_RECORD_FIELDS)
ECO_DRC_C1        = ECO_RECORD_FIELDS.index('DRC_C1')
ECO_DRC_C2        = ECO_RECORD_FIELDS.index('DRC_C2')
ECO_DRC_AND       = ECO_RECORD_FIELDS.index('DRC_AND')
ECO_DRC_OR        = ECO_RECORD_FIELDS.index('DRC_OR')

ECO_CL   = 0.95  # this is the level of confidence used in [1]
ECO_BSNR = 5
ECO_BSSS = 1000

IC_field = namedtuple('IC', ['low', 'high'])
class IC:
  def __init__(self, low, high):
    self.confidence_interval = IC_field(low, high)

def recast_UAlb(val):
  """
    Tries to recast <val> as a float. Returns -1.0 if recasting is not possible.
    It must be noted that when recast UAlb is -1.0, the sample will be rejected downstream.
  """

  if(type(val) == float):
    if(np.isnan(val)):
      res = -1.0
    else:
      # assumes all float instances are np.nan, based on manual inspection
      # is counter-examples are found, the code must be updated accordingly
      raise TypeError
  elif(type(val) == str):
    res = float(val.replace(',', ''))
  else:
    # assumes instances are either float or string, based on manual inspection
    # is counter-examples are found, the code must be updated accordingly
    raise TypeError

  return res

def compute_eGFR(SCr, Age, Gender, Race):
  """
  Computes the estimated GFR (glomerular filtration rate) using the CKD-EPI equation described in [1].

  Assumes:
  - SCr    in micrograms per deciliter
  - Age    in years
  - Gender as a string in ["Masculino", "Feminino"]
  - Race   as a string in ["Amarelo", "Branco", "Indigena", "Pardo", "Preta"]
  Returns eGFR in mililiter per minute, normalised per 1.73 m2 of body surface
  """

  if(SCr > 0.0 and Age > 0 and Gender != '' and Race != ''):

    if(Gender == ECO_FEMALE):
      alpha = -0.329
      kappa =  0.7
      beta  = -1.209
    else:
      alpha = -0.411
      kappa =  0.9
      beta  = -1.209

    gender_adj = 1.018 if (Gender == ECO_FEMALE) else 1.000
    race_adj   = 1.159 if (Race   == ECO_BLACK)  else 1.000

    res = (141 *
           min(SCr / kappa, 1) ** alpha *
           max(SCr / kappa, 1) ** beta  *
           (0.993 ** Age)               *
           gender_adj                   *
           race_adj)

  else:
    res = None

  return res

def compute_ACR(UAlb, UCr, UDur, UVol):
  """
  Computes the estimated ACR (albumin-creatinine ratio) using the ACR equation _suggested by_ [2].
  (In other words, authors of [2] did not explicitly expressed the equation we apply here)
  Assumes:
  - UAlb in micrograms per minute,    and should always be larger than 0.0
  - UCr  in micrograms per deciliter, and should always be larger than 0.0
  - UDur in minutes,                  and should always be larger than 0.0
  - UVol in mililiters,               and should always be larger than 0.0
  Returns ACR, in micrograms per gram
  """

  try:
    if(UAlb > 0.0 and UCr > 0.0 and UDur > 0 and UVol > 0):
      res = UAlb * UDur * (1 / UVol) * (1 / UCr) * (1 / 100)
    else:
      res = None
  except TypeError:
    res = None

  return res

def compute_C1(eGFR):
  return eGFR < 60.0

def compute_C2(ACR):
  return ACR >= 30.0

def resample(dataset, feature, ns, ss):
  """
  Resamples the <dataset> to obtain <ns> sets with <ss> original samples.
  This resampling is required by compute_prevalence_ic_bs.
  """

  dataset_cut = [e[feature] for e in dataset]

  if(ss == -1):
    res = [dataset_cut]

  else:

    res = []
    for i in range(ns):
      np.random.seed(ECO_SEED + i % ECO_SEED)
      res.append(np.random.choice(dataset_cut, size = ss))
    np.random.seed(ECO_SEED)
    return res

def compute_prevalence_ic_na(resamples, level):
  """
  Computes the confidence interval of the mean of <resamples>, assuming normality of distributions.
  Assumes <resamples> is an array of Boolean values.
  see https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
  """
  data = []
  for resample in resamples:
    data.append(sum(resample)/len(resample))
  ddof = len(data) - 1
  mu   = np.mean(data)
  sd   = np.std(data, ddof=1)#try with ddof
  ww   = tdist.ppf((1 + level) / 2., ddof)
  res  = IC(mu - ww * sd, mu + ww * sd)
  return res

def compute_prevalence_ic_bs(resamples, level):
  """
  Computes the confidence interval of the mean of <resamples>, using the bootstrap procedure [3].
  Assumes <resamples> is an array of Boolean values.
  """
  data = []
  for resample in resamples:
    data.append(sum(resample)/len(resample))
  res = bootstrap((data,), np.mean, confidence_level = level)
  return res

def main():

  # defines the application parameters
  np.random.seed(ECO_SEED)
  targetpath = ['D:\\', 'Task Stage', 'Task - collabSergio', 'collabSergio', 'datasets']
  filename   = 'ELSA_Prevalence.tsv'

  # loads the dataset
  df = pd.read_csv(join(*targetpath, filename), sep='\t')
  df.info()

  # obtains eFGR and ACR values for each sample
  accepted = {}
  rejected = {}
  for index, row in df.iterrows():
    (IDELSA, Gender, Age, SCr, UCr, UAlb_ay, UAlb_az, UVol, UDur, Race) = row
    UAlb = recast_UAlb(UAlb_ay)
    eGFR = compute_eGFR(SCr, Age, Gender, Race)
    ACR  = compute_ACR(UAlb, UCr, UDur, UVol)
    if(eGFR is None or ACR is None):
      rejected[IDELSA] = Record(eGFR, ACR, None, None, None, None)
    else:
      C1 = compute_C1(eGFR)
      C2 = compute_C2(ACR)
      accepted[IDELSA] = Record(eGFR, ACR, C1, C2, C1 and C2, C1 or C2)

  # obtains point estimations for different prevalences of interest
  N = len(accepted)
  count_drc_c1  = sum(1 if e.DRC_C1 and not e.DRC_C2 else 0 for e in accepted.values())
  count_drc_c2  = sum(1 if e.DRC_C2 and not e.DRC_C1 else 0 for e in accepted.values())
  count_drc_and = sum(1 if e.DRC_C1 and     e.DRC_C2 else 0 for e in accepted.values())
  count_drc_or  = sum(1 if e.DRC_C1 or      e.DRC_C2 else 0 for e in accepted.values())

  print()
  print('----------------------------------------------------------------------------')
  print('*** POINT ESTIMATION APPROACH ***')
  print('----------------------------------------------------------------------------')

  print()
  print('Number of rejected samples: {0:5d}'.format(len(rejected)))
  print('Number of accepted samples: {0:5d}'.format(len(accepted)))

  print()
  print('Number of individuals with DRC,  exclusively according to Condition 1:   {0:3d}'.format(count_drc_c1))
  print('Number of individuals with DRC,  exclusively according to Condition 2:   {0:3d}'.format(count_drc_c2))
  print('Number of individuals with DRC,  according to both   conditions .....:   {0:3d}'.format(count_drc_and))
  print('Number of individuals with DRC,  according to either conditions .....:   {0:3d}'.format(count_drc_or))

  print()
  print('Prevalence of DRC in the sample, exclusively according to Condition 1: {0:5.3f}'.format(count_drc_c1/N))
  print('Prevalence of DRC in the sample, exclusively according to Condition 2: {0:5.3f}'.format(count_drc_c2/N))
  print('Prevalence of DRC in the sample, according to both   conditions .....: {0:5.3f}'.format(count_drc_and/N))
  print('Prevalence of DRC in the sample, according to either conditions .....: {0:5.3f}'.format(count_drc_or/N))


  # estimates the confidence interval of the parameters of interest in the accepted samples
  sample = [e for e in accepted.values()]

  resamples_drc_c1  = resample(sample, feature = ECO_DRC_C1,  ns = ECO_BSNR, ss = ECO_BSSS)
  resamples_drc_c2  = resample(sample, feature = ECO_DRC_C2,  ns = ECO_BSNR, ss = ECO_BSSS)
  resamples_drc_and = resample(sample, feature = ECO_DRC_AND, ns = ECO_BSNR, ss = ECO_BSSS)
  resamples_drc_or  = resample(sample, feature = ECO_DRC_OR,  ns = ECO_BSNR, ss = ECO_BSSS)

  print()
  print('----------------------------------------------------------------------------')
  print('*** INTERVAL ESTIMATION APPROACH (PARAMETRIC - NORMAL) ***')
  print('----------------------------------------------------------------------------')

  ic_p_hat_c1  = compute_prevalence_ic_na(resamples_drc_c1,  level = ECO_CL)
  ic_p_hat_c2  = compute_prevalence_ic_na(resamples_drc_c2,  level = ECO_CL)
  ic_p_hat_and = compute_prevalence_ic_na(resamples_drc_and, level = ECO_CL)
  ic_p_hat_or  = compute_prevalence_ic_na(resamples_drc_or,  level = ECO_CL)

  print()
  print('Prevalence of DRC in the sample, exclusively according to Condition 1: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_c1.confidence_interval.low, ic_p_hat_c1.confidence_interval.high))
  print('Prevalence of DRC in the sample, exclusively according to Condition 2: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_c2.confidence_interval.low, ic_p_hat_c2.confidence_interval.high))
  print('Prevalence of DRC in the sample, according to both   conditions .....: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_and.confidence_interval.low, ic_p_hat_and.confidence_interval.high))
  print('Prevalence of DRC in the sample, according to either conditions .....: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_or.confidence_interval.low,  ic_p_hat_or.confidence_interval.high))

  print()
  print('----------------------------------------------------------------------------')
  print('*** INTERVAL ESTIMATION APPROACH (NON-PARAMETRIC - BOOTSTRAP) ***')
  print('----------------------------------------------------------------------------')

  ic_p_hat_c1  = compute_prevalence_ic_bs(resamples_drc_c1,  level = ECO_CL)
  ic_p_hat_c2  = compute_prevalence_ic_bs(resamples_drc_c2,  level = ECO_CL)
  ic_p_hat_and = compute_prevalence_ic_bs(resamples_drc_and, level = ECO_CL)
  ic_p_hat_or  = compute_prevalence_ic_bs(resamples_drc_or,  level = ECO_CL)

  print()
  print('Prevalence of DRC in the sample, exclusively according to Condition 1: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_c1.confidence_interval.low, ic_p_hat_c1.confidence_interval.high))
  print('Prevalence of DRC in the sample, exclusively according to Condition 2: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_c2.confidence_interval.low, ic_p_hat_c2.confidence_interval.high))
  print('Prevalence of DRC in the sample, according to both   conditions .....: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_and.confidence_interval.low, ic_p_hat_and.confidence_interval.high))
  print('Prevalence of DRC in the sample, according to either conditions .....: [{0:5.3f}, {1:5.3f}]'.format(ic_p_hat_or.confidence_interval.low,  ic_p_hat_or.confidence_interval.high))

def test():

  rng  = np.random.default_rng()
  dist = norm(loc=2, scale=4)  # our "unknown" distribution
  data = dist.rvs(size=10000, random_state=rng)

  mu_true   = dist.mean()    # the true value of the statistic
  mu_sample = np.mean(data)  # the sample statistic

  data = (data,)  # samples must be in a sequence
  res  = bootstrap(data, np.mean, confidence_level=0.95, random_state=rng)
  print(mu_true, mu_sample, res.confidence_interval)

if __name__ == "__main__":

  main()
  #test()
