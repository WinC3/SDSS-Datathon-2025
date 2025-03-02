import numpy as np
from itertools import combinations_with_replacement

N = 3
vars = []
for i in range(N):
    vars.append("x" + str(i))

terms = []
degree = N + 1
combs = []
for deg in range(degree):
    combs += list(combinations_with_replacement(vars, deg))
valOptName = 'valOpt'

with open('poly_model_func.txt', 'w') as file:
    # function header help
    for i in range(97, 97 + len(combs)):
        file.write(chr(i))
        file.write(', ')
    file.write('\n')
    # unpack independent variables help
    for var in vars:
        file.write(var)
        file.write(', ')
    file.write('\n')
    # valOpt help
    for i in range(97, 97 + len(combs)):
        file.write(valOptName)
        file.write('[' + str(i - 97) + ']')
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

