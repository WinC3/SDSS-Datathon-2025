from itertools import combinations_with_replacement

N = 3 # number of terms
vars = []
for i in range(N):
    vars.append("x" + str(i))

degree = N + 1
combs = []
for deg in range(degree): # give all possible terms from degree 0 to max degree
    combs += list(combinations_with_replacement(vars, deg))
valOptName = 'valOpt'

coefs = []
for perm in list(combinations_with_replacement('abcde', 5)): # generate unique coef names
    toadd = ''
    for elem in perm:
        toadd += elem
    coefs.append(toadd)
print(coefs) # for debug
with open('nonlinear_fit\poly_model_func.txt', 'w') as file:
    # function header help
    for i in range(len(combs)):
        file.write(coefs[i])
        file.write(', ')
    file.write('\n')
    # unpack independent variables help
    for var in vars:
        file.write(var)
        file.write(', ')
    file.write('\n')
    # valOpt help
    for i in range(len(combs)):
        file.write(valOptName)
        file.write('[' + str(i) + ']')
        file.write(', ')
    file.write('\n')
    # combinations
    i = 0
    for comb in combs:
        file.write(coefs[i])
        file.write('*')
        for term in comb:
            file.write(term)
            file.write('*')
        file.write('1') # X1*X2*1, simpler code structure
        file.write('+')
        i += 1

file.close()