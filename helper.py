import numpy as np


def lognormal(mu, sigma):
    return np.random.lognormal(mean=mu, sigma=sigma)

def normal(mu, sigma):
    return np.random.normal(loc=mu, scale=sigma)

def poisson(lmbda):
    return np.random.poisson(lam=lmbda)

def getRandomUniform(low, high):
    return np.random.uniform(low, high)