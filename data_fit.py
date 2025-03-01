import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.optimize import curve_fit

mpl.rcParams['figure.dpi'] = 100

estateData = \
np.loadtxt('load_data.csv', skiprows=1, delimiter=',', unpack=True)

def overall(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return (a + b + c + d + e + f + g + h + i + j + k + l) * x

def polynomial_model(x, a, b, c):
                     #, d, e):
                     #, f, g, h, i, j):
    return a + b*x + c*x**2# + d*x**3 + e*x**4 # + f*x**5 + g*x**6 + h*x**7 + i*x**8 + j*x**9

def findYErr(cov, x):
    total = 0
    for i in range(len(cov)):
        total += np.sqrt(cov[i][i])*x**i

dataTypes = ["beds", "baths", "DEN", "parking", "D_mkt", "building_age", "maint", "price", "new_size", "new_exposure", "new_ward"]
valOpts, valCovs = [], []

price = estateData[7]

maxprice = 0
minprice = 1000000000 # very big num
for val in price:
    maxprice = max(val, maxprice)
    minprice = min(val, minprice)
price_domain = np.linspace(minprice, maxprice, 1000)

for i in range(len(estateData)):
    print(i)
    if (i == 7):
        continue
    valOpt, valCov = curve_fit(polynomial_model, price, estateData[i])
    print(valCov)
    plt.plot(price, estateData[i], ls='', marker='.', label=dataTypes[i])
    plt.errorbar(price_domain, polynomial_model(price_domain, valOpt[0], valOpt[1], valOpt[2]), \
                 yerr=findYErr(valCov, price_domain), fmt='.')#, valOpt[2], valOpt[3], valOpt[4]))
    plt.xlabel('price')
    plt.ylabel(dataTypes[i])
    plt.legend()
    plt.show()
    valOpts.append(valOpt)
    valCovs.append(valCovs)
