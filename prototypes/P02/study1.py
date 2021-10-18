# new environment
# scipy.stats 1.7.0

import numpy as np
rng = np.random.default_rng()
from scipy.stats import norm
dist = norm(loc=2, scale=4)  # our "unknown" distribution
data = dist.rvs(size=100, random_state=rng)

std_true = dist.std()      # the true value of the statistic
print(std_true)
std_sample = np.std(data)  # the sample statistic
print(std_sample)

from scipy.stats import bootstrap
data = (data,)  # samples must be in a sequence
res = bootstrap(data, np.std, confidence_level=0.9, random_state=rng)
print(res.confidence_interval)


