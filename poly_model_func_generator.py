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
    # function header help
    for var in vars:
        file.write(var)
        file.write(', ')
    file.write('\n')
    # combinations
    i = 97 # ascii letter start
    for comb in combs:
        file.write(chr(i))
        file.write('*')
        for term in comb:
            file.write(term)
            file.write('*')
        file.write('1') # X1*X2*1, simpler code structure
        file.write('+')
        i += 1

