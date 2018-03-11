steps = 10
stepsize = size(data)/steps
data.sort()
for i in range(0, streps):
    threshold[i] = data[(i+1)*stepsize]
