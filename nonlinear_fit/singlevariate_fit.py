import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

def pltHelper(plt):
    plt.xlabel(dataTypes[i])
    plt.ylabel('price in millions')
    plt.savefig('monovariate_plots/' + dataTypes[i] + '_plot.png')
    plt.show()

mpl.rcParams['figure.dpi'] = 100

estateData = \
np.loadtxt('load_data.csv', skiprows=1, delimiter=',', unpack=True)

def polynomial_model(x, a, b, c, d):
                     #, e, f, g, h, i, j):
    return a + b*x + c*x**2 + d*x**3 #+ e*x**4 + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9

def findYErr(cov, x):
    total = 0
    for i in range(len(cov)):
        total += np.sqrt(cov[i][i])*x**i

from typing import Tuple

def maxMin(data) -> Tuple[int, int]:
    minV = 100000000000
    maxV = -100000000000
    for val in data:
        minV = min(minV, val)
        maxV = max(maxV, val)
    return (minV, maxV)

dataTypes = ["beds", "baths", "DEN", "parking", "D_mkt", "building_age", "maint", "price", "new_size", "new_exposure", "new_ward"]
valOpts, valCovs = [], []

price = estateData[7]

for i in range(len(estateData)):
    independentVar = estateData[i]
    minVal, maxVal = maxMin(independentVar)
    predDomain = np.linspace(minVal, maxVal, 1000)
    if (i == 7): # price
        continue
    valOpt, valCov = curve_fit(polynomial_model, independentVar, price)
    plt.plot(independentVar, price, ls='', marker='.', label=dataTypes[i])
    plt.errorbar(predDomain, polynomial_model(predDomain, valOpt[0], valOpt[1], valOpt[2], valOpt[3]), \
                 yerr=findYErr(valCov, independentVar), fmt='.')
    plt.title('Prediction plot against actual values for ' + dataTypes[i])
    pltHelper(plt)
    valOpts.append(valOpt)
    valCovs.append(valCovs)
