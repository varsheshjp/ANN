import numpy as np
import csv
from NeuralNet import Network
input=[]
output=[]
classes=['B','R','L']
with open('balance-scale.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    first=True
    for row in csv_reader:
        if first:
            first=False
        else:
            input.append([int(row[0]),int(row[1]),int(row[2]),int(row[3])])
            cl=[0,0,0]
            cl[classes.index(str(row[4]))]=1
            output.append(cl)
X=np.array(input)
Y=np.array(output)
X=X.T
Y=Y.T
print(" X data shape: ",X.shape)
print(" Y data shape: ",Y.shape)
model=Network([X.shape[0],4,4,4,Y.shape[0]],hidden_layers_fun="tanh")
def accu(y,Y):
    y=y.T
    Y=Y.T
    true=0
    for i in range(0,y.shape[0]):
        if np.where(y[i]==np.amax(y[i]))==np.where(Y[i]==np.amax(Y[i])):
            true+=1
    return true/y.shape[0]
model.train(X,Y,0.001,10000,accu,True,True)


