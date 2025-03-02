import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.pyplot as plt


mpl.rcParams['figure.dpi'] = 100

estateData = \
np.loadtxt('load_data1.csv', skiprows=1, delimiter=',', unpack=True)

def poly_model(X, aaaaa, aaaab, aaaac, aaaad, aaaae, aaabb, aaabc, aaabd, aaabe, aaacc, aaacd, aaace, aaadd, aaade, aaaee, aabbb, aabbc, aabbd, aabbe, aabcc): 
    # all components of this function can be found in poly_model_func.txt
    x0, x1, x2 = X
    return aaaaa*1+aaaab*x0*1+aaaac*x1*1+aaaad*x2*1+aaaae*x0*x0*1+aaabb*x0*x1*1+aaabc*x0*x2*1+aaabd*x1*x1*1+aaabe*x1*x2*1+aaacc*x2*x2*1+aaacd*x0*x0*x0*1+aaace*x0*x0*x1*1+aaadd*x0*x0*x2*1+aaade*x0*x1*x1*1+aaaee*x0*x1*x2*1+aabbb*x0*x2*x2*1+aabbc*x1*x1*x1*1+aabbd*x1*x1*x2*1+aabbe*x1*x2*x2*1+aabcc*x2*x2*x2*1

price = estateData[7]

# beds,baths,DEN,parking,D_mkt,building_age,maint,price,new_size,new_exposure,new_ward
X_data = [estateData[0], estateData[6], estateData[8]] # beds maint and newsize. input parameters changed until we found a most accurate one

train_x = []
val_x = []
train_size = int(4*len(estateData[0]) / 5)
for dataColumn in X_data:
    train_x.append(dataColumn[:train_size])
    val_x.append(dataColumn[train_size:])

valOpt, valCov = curve_fit(poly_model, train_x, price[:train_size])

# compute difference with mse = np.mean((predictions - y_val) ** 2)
preds = poly_model(val_x, valOpt[0], valOpt[1], valOpt[2], valOpt[3], valOpt[4], valOpt[5], valOpt[6], valOpt[7], valOpt[8], valOpt[9], valOpt[10], valOpt[11], valOpt[12], valOpt[13], valOpt[14], valOpt[15], valOpt[16], valOpt[17], valOpt[18], valOpt[19])

mse = np.mean((preds - price[train_size:])**2)
acc = np.mean((np.abs(preds - price[train_size:])/price[train_size:])) * 100
print(mse, acc)

print(valOpt)

# valOpt suggests significant variance in x0, x1, x3, and significant covariance in x0 and x1, and x0 and x2
x0 = np.linspace(0, 4, len(price))
x1 = np.linspace(0, 1500, len(price))
x2 = np.linspace(0, 5000, len(price))
X = [x0, x1, x2]

# residuals graph
for x in X:
    plt.plot(x, price)
    plt.show()