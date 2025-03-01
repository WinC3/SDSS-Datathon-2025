import numpy as np
from itertools import combinations_with_replacement

N = 3
vars = []
for i in range(N):
    vars.append("x" + str(i))

terms = []
degree = N
combs = list(combinations_with_replacement(vars, degree))

with open('poly_model_func.txt', 'w') as file:
    for comb in combs:
        for term in comb:
            file.write(term)
            file.write('*')
        file.write('1') # X1*X2*1, simpler code structure
        file.write('+')

