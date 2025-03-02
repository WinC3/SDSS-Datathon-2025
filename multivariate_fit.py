import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement

mpl.rcParams['figure.dpi'] = 100

estateData = \
np.loadtxt('load_data.csv', skiprows=1, delimiter=',', unpack=True)

def poly_model(X, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t):
    x0, x1, x2 = X
    return a*1+b*x0*1+c*x1*1+d*x2*1+e*x0*x0*1+f*x0*x1*1+g*x0*x2*1+h*x1*x1*1+i*x1*x2*1+j*x2*x2*1+k*x0*x0*x0*1+l*x0*x0*x1*1+m*x0*x0*x2*1+n*x0*x1*x1*1+o*x0*x1*x2*1+p*x0*x2*x2*1+q*x1*x1*x1*1+r*x1*x1*x2*1+s*x1*x2*x2*1+t*x2*x2*x2*1

price = estateData[7]

# beds,baths,DEN,parking,D_mkt,building_age,maint,price,new_size,new_exposure,new_ward
X_data = [estateData[8], estateData[0], estateData[6]] # beds maint and newsize

valOpt, valCov = curve_fit(poly_model, X_data, price)

maxprice = 0
minprice = 1000000000 # very big num
for val in price:
    maxprice = max(val, maxprice)
    minprice = min(val, minprice)
price_domain = np.linspace(minprice, maxprice, 1000)

# compute difference with mse = np.mean((predictions - y_val) ** 2)
preds = poly_model(X_data, valOpt[0], valOpt[1], valOpt[2], valOpt[3], valOpt[4], valOpt[5], valOpt[6], valOpt[7], valOpt[8], valOpt[9], valOpt[10], valOpt[11], valOpt[12], valOpt[13], valOpt[14], valOpt[15], valOpt[16], valOpt[17], valOpt[18], valOpt[19])

mse = np.mean((preds - price)**2)
acc = np.mean((np.abs(preds - price)/price)) * 100
print(mse, acc)