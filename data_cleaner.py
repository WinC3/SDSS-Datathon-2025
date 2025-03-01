
with open('real-estate-data.csv', 'r') as file:
    data = file.readlines()

file.close()

with open('pruned-estate-data.csv', 'w') as file:
    for line in data:
        elements = line.split(',')
        add = True
        for i in range(14):
            if (elements[i] == "NA"):
                add = False
        if add:
            file.write(line)