import numpy as np
class Network():
    def __init__(self,Layers_Dim,hidden_layers_fun="tanh"):
        np.random.seed(1)
        self.assertCheck_activation(hidden_layers_fun)
        self.fun=hidden_layers_fun
        self.parameters={}
        self.gradients={}
        self.Layer_length=len(Layers_Dim)
        self.Layer_Dim=Layers_Dim
        for layer in range(1, self.Layer_length):           
            self.parameters["W" + str(layer)] = np.random.randn(Layers_Dim[layer], Layers_Dim[layer - 1]) * 0.01
            self.parameters["b" + str(layer)] = np.zeros((Layers_Dim[layer], 1))
            assert self.parameters["W" + str(layer)].shape == (Layers_Dim[layer], Layers_Dim[layer - 1])
            assert self.parameters["b" + str(layer)].shape == (Layers_Dim[layer], 1)
    
    #activation functions
    def sigmoid(self,x):
        y = 1 / (1 + np.exp(-x))
        return y, x
    def tanh(self,x):
        y = np.tanh(x)
        return y, x
    def relu(self,x):
        y = np.maximum(0, x)
        return y, x
    def leaky_relu(self,x):
        y = np.maximum(0.1 * x, x)
        return y, x
    
    #linear function
    def linear(self,x,w,b):
        v=np.dot(w,x)+b
        inputs=(x,w,b)
        return v,inputs
    def linear_activation(self,x,w,b,Output_layer=False):
        function=self.fun
        if Output_layer:
            function="sigmoid"

        if function == "sigmoid":
            v, linear_inputs = self.linear(x, w, b)
            y, activation_inputs = self.sigmoid(v)
        elif function == "tanh":
            v, linear_inputs = self.linear(x, w, b)
            y, activation_inputs = self.tanh(v)

        elif function == "relu":
            v, linear_inputs = self.linear(x, w, b)
            y, activation_inputs = self.relu(v)

        assert y.shape == (w.shape[0], x.shape[1])

        inputs = (linear_inputs, activation_inputs)
        return y, inputs
    
    #assert functions
    def assertCheck_activation(self,fun):
        assert fun == "sigmoid" or fun == "tanh" or fun == "relu"

    
    def forward(self,X):
        x=X
        inputs=[]
        L = len(self.parameters) // 2
        for layer in range(1,L):
            x,in_1=self.linear_activation(x,self.parameters["W" + str(layer)],self.parameters["b" + str(layer)],False)
            inputs.append(in_1)
        x,in_2=self.linear_activation(x,self.parameters["W" + str(L)],self.parameters["b" + str(L)],True)
        inputs.append(in_2)
        assert x.shape == (self.Layer_Dim[-1],X.shape[1])
        return x,inputs
    #cost function
    def cost_function(self,x, y):
        m = y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(y, np.log(x)) + np.multiply(1 - y, np.log(1 - x)))
        return cost

    #gradient functions
    def sigmoid_gradient(self,delta_x, v):
        y, v = self.sigmoid(v)
        dZ = delta_x * y * (1 - y)
        return dZ
    def tanh_gradient(self,delta_x, v):
        y, v = self.tanh(v)
        dZ = delta_x * (1 - np.square(y))
        return dZ
    def relu_gradient(self,delta_x, v):
        y, v = self.relu(v)
        dZ = np.multiply(delta_x, np.int64(y > 0))
        return dZ
    def linear_backword(self,dZ, inputs):
        x, W, b = inputs
        m = x.shape[1]

        dW = (1 / m) * np.dot(dZ, x.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        dx = np.dot(W.T, dZ)

        assert dx.shape == x.shape
        assert dW.shape == W.shape
        assert db.shape == b.shape
        return dx, dW, db

    def linear_activation_backward(self,dx, inputs,Output_layer=False):
        linear_inputs, activation_inputs = inputs
        function=self.fun
        if Output_layer:
            function="sigmoid"

        if function == "sigmoid":
            dZ = self.sigmoid_gradient(dx, activation_inputs)
            dx_prev_layer, dW, db = self.linear_backword(dZ, linear_inputs)

        elif function == "tanh":
            dZ = self.tanh_gradient(dx, activation_inputs)
            dx_prev_layer, dW, db = self.linear_backword(dZ, linear_inputs)

        elif function == "relu":
            dZ = self.relu_gradient(dx, activation_inputs)
            dx_prev_layer, dW, db = self.linear_backword(dZ, linear_inputs)
        return dx_prev_layer, dW, db
    
    def backward(self,y, Y,inputs):
        Y = Y.reshape(y.shape)
        delta_y = np.divide(y - Y, np.multiply(y, 1 - y))
        print(delta_y.shape)
        L=len(inputs)
        self.gradients["dA" + str(L - 1)], self.gradients["dW" + str(L)], self.gradients["db" + str(L)] = self.linear_activation_backward(delta_y, inputs[L - 1], True)
        for l in range(L - 1, 0, -1):
            current_input = inputs[l - 1]
            self.gradients["dA" + str(l - 1)], self.gradients["dW" + str(l)], self.gradients["db" + str(l)] = self.linear_activation_backward(self.gradients["dA" + str(l)], current_input,False)
    
    def update_parameters(self):
        L = len(self.parameters) // 2
        for l in range(1, L + 1):
            self.parameters["W" + str(l)] = self.parameters["W" + str(l)] - self.learning_rate * self.gradients["dW" + str(l)]
            self.parameters["b" + str(l)] = self.parameters["b" + str(l)] - self.learning_rate * self.gradients["db" + str(l)]
    def accuracy(self,function,y,Y):
        return function(y,Y)
    def train(self,X,Y,learning_rate,epoch,accuracy_function,print_accuracy=True,print_cost=True):
        self.learning_rate=learning_rates
        assert X.shape[0]==self.Layer_Dim[0]
        assert Y.shape[0]==self.Layer_Dim[-1]
        for i in range(epoch):
                
            y,inputs=self.forward(X)
            cost=self.cost_function(y,Y)
            self.backward(y,Y,inputs)
            self.update_parameters()
            print("epoch : {0}",i+1,end=" ")
            if print_accuracy:
                accuracy=self.accuracy(accuracy_function,y,Y)
                print(" acc: {0}",accuracy,end=" ")
            if print_cost:
                print(" cost: {0}",cost,end=" ")
            print("")
    def predict(self,x):
        y,inputs=self.forward(x)
        return y
