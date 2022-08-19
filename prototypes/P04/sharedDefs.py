import re
import os
import sys
import codecs
import pickle
import numpy as np

from copy             import copy
from datetime         import datetime
from collections      import defaultdict
from numpy.random     import seed, random, randint, choice, get_state
from numpy.linalg     import norm
#from shapely.geometry import Polygon #Point,

RAD  = 1.0
PRGS = 23
TOLERANCE = 1E-9
ECO_DATETIME_FMT = '%Y%m%d%H%M%S'

#--------------------------------------------------------------------------------------------------
# General purpose definitions - I/O interfaces used in logging and serialisation
#--------------------------------------------------------------------------------------------------

# buffer where all tsprint messages are stored
LogBuffer = []

def stimestamp():
  return(datetime.now().strftime(ECO_DATETIME_FMT))

def stimediff(finishTs, startTs):
  return str(datetime.strptime(finishTs, ECO_DATETIME_FMT) - datetime.strptime(startTs, ECO_DATETIME_FMT))

def tsprint(msg, verbose=True):
  buffer = '[{0}] {1}'.format(stimestamp(), msg)
  #buffer = '{0}'.format(msg)
  if(verbose):
    print(buffer)
  LogBuffer.append(buffer)

def resetLog():
  LogBuffer = []

def saveLog(filename):
  saveAsText('\n'.join(LogBuffer), filename)

def headerfy(mask):
  res = re.sub('\:\d+\.\d+f', '', mask)
  res = re.sub('\:\d+d', '', res)
  return res

def saveAsText(content, filename, _encoding='utf-8'):
  f = codecs.open(filename, 'w', encoding=_encoding)
  f.write(content)
  f.close()

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

#-------------------------------------------------------------------------------------------------------------------------------------------
# Problem-specific definitions: polygon manipulation
#-------------------------------------------------------------------------------------------------------------------------------------------

def area(T):
  """
  T is an (d,2)-array with the rectangular coordinates of the vertices of a polygon.
  This returns the area of such polygon.
  """
  (_x, _y) = (0, 1)
  R  = np.roll(np.eye(T.shape[0]), 1, axis=1)
  T_ = R.dot(T)
  a  = .5 * (T[:,_x].dot(T_[:,_y]) - T_[:,_x].dot(T[:,_y]))
  return a

def areap(beta):
  """
  beta is a (d,)-array of [0,1]-magnitudes applied to the base axes that form a polygon
  returns the area of such polygon
  """
  d = len(beta)
  dtheta = 2 * np.pi / d
  acc = 0.0
  for i in range(d):
    j = (i+1) % d
    acc += beta[i] * beta[j]
  return .5 * np.sin(dtheta) * acc

def meetat(t_i, t_j, p_i, p_j):
  """
  t_i and t_j are (x,y)-coordinates of two points that determine some line T
  p_i and p_j are (x,y)-coordinates of two points that determine some line P
  returns the (x,y)-coordinates of the point w where the lines P and T cross
  """
  (_x, _y) = (0, 1)
  ap = (p_j[_y] - p_i[_y])/(p_j[_x] - p_i[_x])
  at = (t_j[_y] - t_i[_y])/(t_j[_x] - t_i[_x])
  bp = p_i[_y] - ap * p_i[_x]
  bt = t_i[_y] - at * t_i[_x]
  wx = (bt - bp)/(ap - at)
  wy = ap * wx + bp

  return np.array([wx, wy])

def meetatp(beta_i, beta_j, alpha_i, alpha_j, theta_i, theta_j):

  #xnum = (+ alpha_i * alpha_j * beta_i * np.cos(theta_i)
  #        - alpha_i * alpha_j * beta_j * np.cos(theta_j)
  #        - alpha_i * beta_i  * beta_j * np.cos(theta_i)
  #        + alpha_j * beta_i  * beta_j * np.cos(theta_j)
  #       )
  #
  #ynum = (- alpha_i * alpha_j * beta_i * np.sin(theta_i)
  #        + alpha_i * alpha_j * beta_j * np.sin(theta_j)
  #        + alpha_i * beta_i  * beta_j * np.sin(theta_i)
  #        - alpha_j * beta_i  * beta_j * np.sin(theta_j)
  #      )

  mags = np.array([ + alpha_i * alpha_j * beta_i,
                    - alpha_i * alpha_j * beta_j,
                    - alpha_i * beta_i  * beta_j,
                    + alpha_j * beta_i  * beta_j
                  ])

  args = np.array([theta_i, theta_j, theta_i, theta_j])

  xnum = sum(mags * np.cos(args))
  ynum = sum(mags * np.sin(args))
  den = alpha_j * beta_i - alpha_i * beta_j

  return (xnum/den, ynum/den)


def intersection(T, P):
  """
  T and P are (d,2)-arrays with (x,y)-coordinates determining two polygons
  returns a (d...2d, 2)-np.array that defines the polygon formed by the intersection
  between T and P
  """
  d = T.shape[0] # points are rows within T,P matrices
  V = []
  for i in range(d):
    V.append(P[i] if norm(P[i]) < norm(T[i]) else T[i])
    j = (i + 1) % d
    if( (norm(P[i]) - norm(T[i])) * (norm(P[j])- norm(T[j])) < 0):
      V.append(meetat(T[i], T[j], P[i], P[j]))
  return(np.array(V))

def intersectionp(beta, alpha):
  """
  beta  is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon T
  alpha is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon P
  """
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta  = [i * dtheta for i in range(d)]

  V = []
  for i in range(d):
    if(alpha[i] < beta[i]):
      V.append([alpha[i] * np.cos(theta[i]), alpha[i] * np.sin(theta[i])])
    else:
      V.append([ beta[i] * np.cos(theta[i]),  beta[i] * np.sin(theta[i])])
    j = (i + 1) % d
    if((alpha[i] - beta[i]) * (alpha[j] - beta[j]) < 0):
      V.append(meetatp(beta[i], beta[j], alpha[i], alpha[j], theta[i], theta[j]))
  return(np.array(V))

def gradareap(beta):
  """
  beta is a (d,)-array of [0,1]-magnitudes applied to the base axes that form a polygon
  returns the gradient of the area of such polygon with respect to these magnitudes
  """
  d = len(beta)
  dtheta = 2 * np.pi / d
  grad = np.array([.5 * np.sin(dtheta) * (beta[i-1] + beta[(i+1) % d]) for i in range(d)])
  return grad

def gradintersection(alpha, beta, truth = None, delta = None):
  """
  alpha is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon P
  beta  is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon T
  returns the gradient of the area of the intersection between T and P with respect to the
  magnitudes of T
  """

  # this is an approximation
  #gamma = np.minimum(alpha, beta)
  #grad  = gradareap(gamma)
  #return grad

  ## this is a better approximation
  #d = beta.shape[0]
  #gamma = np.minimum(alpha, beta)
  #aux = np.array([1.0 if beta[i] <= alpha[i] else 0.0 for i in range(d)])
  #grad  = gradareap(gamma) * aux
  #return grad

  # this is the exact solution
  d = beta.shape[0]
  dtheta = 2 * np.pi / d

  crossings = []
  for i in range(d):
    j = (i + 1) % d
    crossings.append(1 if ((beta[i] - alpha[i]) * (beta[j] - alpha[j]) < 0) else 0)

  labels = []
  grad = np.zeros(d)
  for i in range(d):
    j = (i + 1) % d
    h = i - 1

    # only 4 possible configs for each axis
    if (crossings[i] == 0):

      if (crossings[h] == 0):

        # no crossings
        if beta[i] >= alpha[i]:
          label = 'A'
          grad[i] = 0.0
        else:
          label = 'a'
          grad[i] = .5 * np.sin(dtheta) * (beta[h] + beta[j])

      else:

        # single crossing, previous axis (CCW)
        if beta[i] > alpha[i]:
          label = 'C'
          num = beta[h] * alpha[i] ** 2 * (alpha[h] - beta[h]) ** 2 * np.sin(dtheta)
          den = 2 * (alpha[h] * beta[i] - alpha[i] * beta[h]) ** 2
          grad[i] = np.clip(num/den, -RAD, RAD)
        else:
          label = 'c'
          num = ( -     alpha[i] ** 2 * beta[h] ** 2 * beta[j]  * np.sin(-dtheta)
                  + 2 * alpha[i] ** 2 * beta[h] ** 2 * alpha[h] * np.sin( dtheta)
                  -     alpha[i] ** 2 * beta[h] * alpha[h] ** 2 * np.sin( dtheta)
                  - 2 * alpha[i] * beta[h] ** 2 * beta[i] * alpha[h] * np.sin( dtheta)
                  + 2 * alpha[i] * beta[h] * beta[i] * beta[j] * alpha[h] * np.sin(-dtheta)
                  +     beta[h] * beta[i] ** 2 * alpha[h] ** 2 * np.sin( dtheta)
                  -     beta[i] ** 2 * beta[j] * alpha[h] ** 2 * np.sin(-dtheta)
                )
          den = 2 * (alpha[i] * beta[h] - beta[i] * alpha[h]) ** 2
          grad[i] = np.clip(num/den, -RAD, RAD)

    else:

      if (crossings[h] == 0):

        # single crossing, next axis (CCW)
        if beta[i] > alpha[i]:
          label = 'B'
          num = -beta[j] * alpha[i] ** 2 * (alpha[j] - beta[j]) ** 2 * np.sin(-dtheta)
          den = 2 * (alpha[j] * beta[i] - alpha[i] * beta[j]) ** 2
          grad[i] = np.clip(num/den, -RAD, RAD)
        else:
          label = 'b'
          num = ( - 2 * alpha[i] ** 2 * beta[j] ** 2 * alpha[j]           * np.sin(-dtheta)
                  +     alpha[i] ** 2 * beta[j] ** 2 *  beta[h]           * np.sin( dtheta)
                  +     alpha[i] ** 2 * beta[j]      * alpha[j] ** 2      * np.sin(-dtheta)
                  + 2 * alpha[i] * beta[j] ** 2 *  beta[i] * alpha[j]     * np.sin(-dtheta)
                  - 2 * alpha[i] * beta[j] * beta[i] * alpha[j] * beta[h] * np.sin( dtheta)
                  -      beta[j] * beta[i] ** 2 *  alpha[j] ** 2          * np.sin(-dtheta)
                  +      beta[i] ** 2 * alpha[j] ** 2 * beta[h]           * np.sin( dtheta)
                )

          den = 2 * (alpha[i] * beta[j] - beta[i] * alpha[j]) ** 2
          grad[i] = np.clip(num/den, -RAD, RAD)

      else:

        # two crossings
        if beta[i] > alpha[i]:
          label = 'D'
          num =   - alpha[i]** 2 * (
              +     alpha[h]**2 * alpha[j]**2 * beta[i]**2 * beta[j] * np.sin(-dtheta)
              -     alpha[h]**2 * alpha[j]**2 * beta[i]**2 * beta[h] * np.sin( dtheta)
              - 2 * alpha[h]**2 * alpha[j] * beta[i]**2 * beta[j]**2 * np.sin(-dtheta)
              + 2 * alpha[h]**2 * alpha[j] * beta[i] * alpha[i] * beta[j] * beta[h] * np.sin( dtheta)
              +     alpha[h]**2 * beta[i]**2 * beta[j]**3 * np.sin(-dtheta)
              -     alpha[h]**2 * alpha[i]**2 * beta[j]**2 * beta[h] * np.sin( dtheta)
              + 2 * alpha[h] * alpha[j]**2 * beta[i]**2 * beta[h]**2 * np.sin( dtheta)
              - 2 * alpha[h] * alpha[j]**2 * beta[i] * alpha[i] * beta[j] * beta[h] * np.sin(-dtheta)
              + 4 * alpha[h] * alpha[j] * beta[i] * alpha[i] * beta[j]**2 * beta[h] * np.sin(-dtheta)
              - 4 * alpha[h] * alpha[j] * beta[i] * alpha[i] * beta[j] * beta[h]**2 * np.sin( dtheta)
              - 2 * alpha[h] * beta[i] * alpha[i] * beta[j]**3 * beta[h] * np.sin(-dtheta)
              + 2 * alpha[h] * alpha[i]**2 * beta[j]**2 * beta[h]**2 * np.sin( dtheta)
              -     alpha[j]**2 * beta[i]**2 * beta[h]**3 * np.sin( dtheta)
              +     alpha[j]**2 * alpha[i]**2 * beta[j] * beta[h]**2 * np.sin(-dtheta)
              + 2 * alpha[j] * beta[i] * alpha[i] * beta[j] * beta[h]**3 * np.sin( dtheta)
              - 2 * alpha[j] * alpha[i]**2 * beta[j]**2 * beta[h]**2 * np.sin(-dtheta)
              +     alpha[i]**2 * beta[j]**3 * beta[h]**2 * np.sin(-dtheta)
              -     alpha[i]**2 * beta[j]**2 * beta[h]**3 * np.sin( dtheta)
              )
          den =  2 * (alpha[h] * beta[i] - alpha[i] * beta[h])**2 * (alpha[j] * beta[i] - alpha[i] * beta[j])**2
          grad[i] = np.clip(num/den, -RAD, RAD)

        else:
          label = 'd'
          num = - (alpha[i] - beta[i]) * (
                   + 2 * alpha[i] ** 3 * beta[h] ** 2 * beta[j] ** 2 * alpha[j] * np.sin(-dtheta)
                   - 2 * alpha[i] ** 3 * beta[h] ** 2 * beta[j] ** 2 * alpha[h] * np.sin( dtheta)
                   -     alpha[i] ** 3 * beta[h] ** 2 * beta[j] * alpha[j] ** 2 * np.sin(-dtheta)
                   +     alpha[i] ** 3 * beta[h] * beta[j] ** 2 * alpha[h] ** 2 * np.sin( dtheta)
                   -     alpha[i] ** 2 * beta[h] ** 2 * beta[j] * beta[i] * alpha[j] ** 2 * np.sin(-dtheta)
                   + 4 * alpha[i] ** 2 * beta[h] ** 2 * beta[j] * beta[i] * alpha[j] * alpha[h] * np.sin( dtheta)
                   - 4 * alpha[i] ** 2 * beta[h] * beta[j] ** 2 * beta[i] * alpha[j] * alpha[h] * np.sin(-dtheta)
                   +     alpha[i] ** 2 * beta[h] * beta[j] ** 2 * beta[i] * alpha[h] ** 2 * np.sin( dtheta)
                   + 2 * alpha[i] ** 2 * beta[h] * beta[j] * beta[i] * alpha[j] ** 2 * alpha[h] * np.sin(-dtheta)
                   - 2 * alpha[i] ** 2 * beta[h] * beta[j] * beta[i] * alpha[j] * alpha[h] ** 2 * np.sin( dtheta)
                   - 2 * alpha[i] * beta[h] ** 2 * beta[i] ** 2 * alpha[j] ** 2 * alpha[h] * np.sin( dtheta)
                   + 2 * alpha[i] * beta[h] * beta[j] * beta[i] ** 2 * alpha[j] ** 2 * alpha[h] * np.sin(-dtheta)
                   - 2 * alpha[i] * beta[h] * beta[j] * beta[i] ** 2 * alpha[j] * alpha[h] ** 2 * np.sin( dtheta)
                   +     alpha[i] * beta[h] * beta[i] ** 2 * alpha[j] ** 2 * alpha[h] ** 2 * np.sin( dtheta)
                   + 2 * alpha[i] * beta[j] ** 2 * beta[i] ** 2 * alpha[j] * alpha[h] ** 2 * np.sin(-dtheta)
                   -     alpha[i] * beta[j] * beta[i] ** 2 * alpha[j] ** 2 * alpha[h] ** 2 * np.sin(-dtheta)
                   +      beta[h] * beta[i] ** 3 * alpha[j] ** 2 * alpha[h] ** 2 * np.sin( dtheta)
                   -      beta[j] * beta[i] ** 3 * alpha[j] ** 2 * alpha[h] ** 2 * np.sin(-dtheta)
                  )
          den = 2 * (alpha[i] * beta[h] - beta[i] * alpha[h]) ** 2 * (alpha[i] * beta[j] - beta[i] * alpha[j]) ** 2
          grad[i] = np.clip(num/den, -RAD, RAD)
          #try:
          #  grad[i] = num/den
          #except Warning:
          #  if(num == 0.0 and den == 0.0):
          #    grad[i] = 0.0
          #  else:
          #    print(num,den)
          #  pass

    labels.append(label)

  return grad, labels

def intertype(alpha, beta):

  """
  alpha is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon P
  beta  is a (d,)-array of [0,1]-magnitudes applied to the base axes that form some polygon T
  """
  d = beta.shape[0]
  crossings = []
  for i in range(d):
    j = (i + 1) % d
    crossings.append(0 if (beta[i] - alpha[i]) * (beta[j] - alpha[j]) > 0 else 1)

  labels = []
  for i in range(d):
    h = i - 1

    # only 4 possible configs for each axis
    if (crossings[i] == 0):
      if (crossings[h] == 0):
        labels.append('A' if beta[i] > alpha[i] else 'a')
      else:
        labels.append('c' if beta[i] > alpha[i] else 'C')
    else:
      if (crossings[h] == 0):
        labels.append('B' if beta[i] > alpha[i] else 'b')
      else:
        labels.append('D' if beta[i] > alpha[i] else 'd')

  return labels

def CoM(beta):
  # center of mass of a polygon t
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta = np.array([k * dtheta for k in range(d)])

  x = np.mean([beta[k] * np.cos(theta[k]) for k in range(d)], dtype=np.float64)
  y = np.mean([beta[k] * np.sin(theta[k]) for k in range(d)], dtype=np.float64)
  return np.array([x,y])

def CoG(beta):
  # center of gravity of polygon t (this is how Shapely computes the centroid of a Polygon)
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta = np.array([k * dtheta for k in range(d)])

  cx = 0
  cy = 0
  ar = 0
  for i in range(d):
    j = (i+1)%d
    x_i = beta[i] * np.cos(theta[i])
    x_j = beta[j] * np.cos(theta[j])
    y_i = beta[i] * np.sin(theta[i])
    y_j = beta[j] * np.sin(theta[j])
    sl = (x_i * y_j - x_j * y_i)
    ar += sl
    cx += (x_i + x_j) * sl
    cy += (y_i + y_j) * sl
  return np.array([cx/(3*ar), cy/(3*ar)])

def DCoM(alpha, beta):
  # distance between the centers of mass of polygons p and t
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta = np.array([k * dtheta for k in range(d)])
  v = CoM(beta) - CoM(alpha)
  return v.dot(v)

def gradDCoM(alpha, beta):
  # distance between the centers of mass of polygons p and t
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta = np.array([k * dtheta for k in range(d)])
  M = []
  for k in range(d):
    aux = sum([(beta[j] - alpha[j]) * np.cos(theta[k] - theta[j]) for j in range(d)])
    M.append(aux)
  return 2/d**2 * np.array(M)

def DCoG(alpha, beta):
  # distance between the centers of gravity of polygons p and t
  d = len(beta)
  dtheta = 2 * np.pi / d
  theta = np.array([k * dtheta for k in range(d)])
  v = CoG(beta) - CoG(alpha)
  return v.dot(v)


#--------------------------------------------------------------------------------------------------
# These are helpers to define polygons using cartesian or polar coordinates
#--------------------------------------------------------------------------------------------------

def polar2coord(r, theta):
  x = r * np.cos(theta)  # the x-coord of the vertice that corresponds to the current score
  y = r * np.sin(theta)  # the y-coord of the vertice that corresponds to the current score
  return (x, y)

def scores2coords(scores, tomains, ulimits):

  nd = len(tomains)     # number of domains
  ra = 2 * np.pi / nd   # angle between two axes, in radians
  axes  = [(tomains[i], i * ra) for i in range(nd)] # ~ [(domain LABEL, axis angle), ...]

  # converts the scores to the cartesian coordinates describing the vertices of a regular nd-polygon
  L = []
  for (domain, theta) in axes:
    r = scores[domain] / ulimits[domain] # scales the score of the current domain to the [0, 1] interval
    L.append(polar2coord(r, theta))

  return L

def coords2poly(coords):
  return Polygon(coords)

