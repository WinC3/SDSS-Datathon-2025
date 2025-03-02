import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl

mpl.rcParams['figure.dpi'] = 100

estateData = \
np.loadtxt('load_data1.csv', skiprows=1, delimiter=',', unpack=True)

def poly_model(X, aaaaa, aaaab, aaaac, aaaad, aaaae, aaabb, aaabc, aaabd, aaabe, aaacc, aaacd, aaace, aaadd, aaade, aaaee, aabbb, aabbc, aabbd, aabbe, aabcc, aabcd, aabce, aabdd, aabde, aabee, aaccc, aaccd, aacce, aacdd, aacde, aacee, aaddd, aadde, aadee, aaeee, abbbb, abbbc, abbbd, abbbe, abbcc, abbcd, abbce, abbdd, abbde, abbee, abccc, abccd, abcce, abcdd, abcde, abcee, abddd, abdde, abdee, abeee, acccc, acccd, accce, accdd, accde, accee, acddd, acdde, acdee, aceee, adddd, addde, addee, adeee, aeeee, bbbbb, bbbbc, bbbbd, bbbbe, bbbcc, bbbcd, bbbce, bbbdd, bbbde, bbbee, bbccc, bbccd, bbcce, bbcdd):
    # generated from poly_model_func_generator.py
    x0, x1, x2, x3, x4, x5 = X
    return aaaaa*1+aaaab*x0*1+aaaac*x1*1+aaaad*x2*1+aaaae*x3*1+aaabb*x4*1+aaabc*x5*1+aaabd*x0*x0*1+aaabe*x0*x1*1+aaacc*x0*x2*1+aaacd*x0*x3*1+aaace*x0*x4*1+aaadd*x0*x5*1+aaade*x1*x1*1+aaaee*x1*x2*1+aabbb*x1*x3*1+aabbc*x1*x4*1+aabbd*x1*x5*1+aabbe*x2*x2*1+aabcc*x2*x3*1+aabcd*x2*x4*1+aabce*x2*x5*1+aabdd*x3*x3*1+aabde*x3*x4*1+aabee*x3*x5*1+aaccc*x4*x4*1+aaccd*x4*x5*1+aacce*x5*x5*1+aacdd*x0*x0*x0*1+aacde*x0*x0*x1*1+aacee*x0*x0*x2*1+aaddd*x0*x0*x3*1+aadde*x0*x0*x4*1+aadee*x0*x0*x5*1+aaeee*x0*x1*x1*1+abbbb*x0*x1*x2*1+abbbc*x0*x1*x3*1+abbbd*x0*x1*x4*1+abbbe*x0*x1*x5*1+abbcc*x0*x2*x2*1+abbcd*x0*x2*x3*1+abbce*x0*x2*x4*1+abbdd*x0*x2*x5*1+abbde*x0*x3*x3*1+abbee*x0*x3*x4*1+abccc*x0*x3*x5*1+abccd*x0*x4*x4*1+abcce*x0*x4*x5*1+abcdd*x0*x5*x5*1+abcde*x1*x1*x1*1+abcee*x1*x1*x2*1+abddd*x1*x1*x3*1+abdde*x1*x1*x4*1+abdee*x1*x1*x5*1+abeee*x1*x2*x2*1+acccc*x1*x2*x3*1+acccd*x1*x2*x4*1+accce*x1*x2*x5*1+accdd*x1*x3*x3*1+accde*x1*x3*x4*1+accee*x1*x3*x5*1+acddd*x1*x4*x4*1+acdde*x1*x4*x5*1+acdee*x1*x5*x5*1+aceee*x2*x2*x2*1+adddd*x2*x2*x3*1+addde*x2*x2*x4*1+addee*x2*x2*x5*1+adeee*x2*x3*x3*1+aeeee*x2*x3*x4*1+bbbbb*x2*x3*x5*1+bbbbc*x2*x4*x4*1+bbbbd*x2*x4*x5*1+bbbbe*x2*x5*x5*1+bbbcc*x3*x3*x3*1+bbbcd*x3*x3*x4*1+bbbce*x3*x3*x5*1+bbbdd*x3*x4*x4*1+bbbde*x3*x4*x5*1+bbbee*x3*x5*x5*1+bbccc*x4*x4*x4*1+bbccd*x4*x4*x5*1+bbcce*x4*x5*x5*1+bbcdd*x5*x5*x5*1

price = estateData[7]

# beds,baths,DEN,parking,D_mkt,building_age,maint,price,new_size,new_exposure,new_ward
X_data = [estateData[0], estateData[1], estateData[2], estateData[3], estateData[6], estateData[8]] # beds maint and newsize

train_x = []
train_size = int(4*len(estateData[0]) / 5) # train on 80% of dataset
for dataColumn in X_data:
    train_x.append(dataColumn[:train_size])

valOpt, valCov = curve_fit(poly_model, train_x, price[:train_size])

# compute difference with mse = np.mean((predictions - y_val) ** 2)
preds = poly_model(X_data, valOpt[0], valOpt[1], valOpt[2], valOpt[3], valOpt[4], valOpt[5], valOpt[6], valOpt[7], valOpt[8], valOpt[9], valOpt[10], valOpt[11], valOpt[12], valOpt[13], valOpt[14], valOpt[15], valOpt[16], valOpt[17], valOpt[18], valOpt[19], valOpt[20], valOpt[21], valOpt[22], valOpt[23], valOpt[24], valOpt[25], valOpt[26], valOpt[27], valOpt[28], valOpt[29], valOpt[30], valOpt[31], valOpt[32], valOpt[33], valOpt[34], valOpt[35], valOpt[36], valOpt[37], valOpt[38], valOpt[39], valOpt[40], valOpt[41], valOpt[42], valOpt[43], valOpt[44], valOpt[45], valOpt[46], valOpt[47], valOpt[48], valOpt[49], valOpt[50], valOpt[51], valOpt[52], valOpt[53], valOpt[54], valOpt[55], valOpt[56], valOpt[57], valOpt[58], valOpt[59], valOpt[60], valOpt[61], valOpt[62], valOpt[63], valOpt[64], valOpt[65], valOpt[66], valOpt[67], valOpt[68], valOpt[69], valOpt[70], valOpt[71], valOpt[72], valOpt[73], valOpt[74], valOpt[75], valOpt[76], valOpt[77], valOpt[78], valOpt[79], valOpt[80], valOpt[81], valOpt[82], valOpt[83])

mse = np.mean((preds - price)**2)
acc = np.mean((np.abs(preds - price)/price)) * 100
print(mse, acc)

avgE = np.mean(preds - price)
print(avgE) # average error