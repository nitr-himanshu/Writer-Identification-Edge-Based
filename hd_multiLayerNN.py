import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


class multilayerNN:

    #activation functions
    def reluFunc(x):
        if (x > 0):
            return x
        else:
            return 0


#vectorizing the reluFunction so that it can be applied to each element in an array or matrix

    ReLU = np.vectorize(reluFunc)

    def deltaReluFunc(x):
        if (x > 0):
            return 1
        else:
            return 0

    deltaReLU = np.vectorize(deltaReluFunc)

    def softmax(x):
        return (np.exp(x) / np.sum(np.exp(x)))

    #loss function
    def cross_entropy(y, y_):
        loss = 0
        for i in range(len(y_)):
            loss += -y[i][0] * np.log(y_[i][0])
        return loss

    #creating intial weight and biases
    def create_weight(dimensions):
        n_layer = len(dimensions) - 1
        weight = list()
        for i in range(n_layer):
            x = np.random.random((dimensions[i], dimensions[i + 1]), )
            weight.append(x)
        return weight

    def create_bias(dimensions):
        n_layer = len(dimensions) - 1
        bias = list()
        for i in range(n_layer):
            x = np.random.random((dimensions[i + 1], 1))
            bias.append(x)
        return bias


    def indexOfMax(arr):
        arr = list(arr)
        return arr.index(max(arr))

    def report(actual,predicted):
        print("Classification report")
        print(classification_report(actual, predicted))
        print("Accuracy ==> ", accuracy_score(actual,predicted)*100)

    def __init__(self, dimension, alpha=0.005, epoch=50):
        self.dims = dimension
        self.alpha = alpha
        self.epoch_count = epoch
        self.no_of_layer = len(self.dims) - 1
        self.no_of_feature = dimension[0]
        self.no_of_class = dimension[-1]
        self.W = multilayerNN.create_weight(self.dims)
        self.B = multilayerNN.create_bias(self.dims)

    def forwardPass(self, sample):
        #to store output at each layer
        netInputList = list()
        inputList = list()

        for i in range(self.no_of_layer):
            inputList.append(sample)
            netInput = np.matmul(self.W[i].T, sample) + self.B[i]
            netInputList.append(netInput)
            #for last layer apply softmax, rest all layer ReLU
            if (i != self.no_of_layer - 1):
                op = multilayerNN.ReLU(netInput)
            else:
                netInput = netInput.reshape((len(netInput)))
                netInput = netInput / 100
                op = multilayerNN.softmax(netInput)
            #current output is input for next layer
            sample = op.reshape((len(op), 1))
        finalOP = sample
        return [inputList, netInputList, finalOP]


    def plot_decision_boundary(self, X, y):
        x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
        y_min, y_max = X[ :,1].min() - 1, X[ :,1].max() + 1
        h = 0.1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        mesh_data = np.c_[xx.ravel(), yy.ravel()]
        z = self.predict(mesh_data)
        z = z.reshape(xx.shape)
        plt.contourf(xx, yy, z, cmap=plt.cm.Spectral)
        plt.ylabel('x2')
        plt.xlabel('x1')
        plt.scatter(X[ :,0], X[ :,1], c=y.ravel(), cmap=plt.cm.Spectral)
        return plt

    def train(self, data, target):
        title = ""
        print("Training network")
        print("Epoch ==> ")
        lossList = list()
        epochList = list()
        for e in range(self.epoch_count):
            if((e+1)%1 == 0):
                print(e+1,end = " ")
            loss = 0
            for i in range(len(data)):
                sample = data[i]
                actual_class = target[i]
                sample = sample.reshape((self.no_of_feature, 1))

                #output of each layer in forward pass
                inputList, netInputList, pred_prob = self.forwardPass(sample)
                actual_prob = [0] * self.no_of_class
                actual_prob[actual_class] = 1
                actual_prob = np.array(actual_prob)
                actual_prob = actual_prob.reshape((self.no_of_class, 1))

                #computing senstivity
                S = list()

                #for last layer
                s = pred_prob - actual_prob
                loss += multilayerNN.cross_entropy(actual_prob,pred_prob)
                S.append(s)
                #for rest layers, back propagating the sensivity
                #because index start from 0 hence last layer is no_of_layer - 1
                #and second last layer is no of layer - 2
                #from second last layer to first layer we have to calculate senstivity
                for i in range(self.no_of_layer - 2, -1, -1):
                    s = np.matmul(self.W[i + 1], S[0])
                    t = multilayerNN.deltaReLU(netInputList[i])
                    s = t * s
                    S.insert(0, s)

                #updating weight and bias
                for i in range(self.no_of_layer):
                    self.W[i] = self.W[i] - self.alpha * np.matmul(inputList[i], S[i].T)
                    self.B[i] = self.B[i] - self.alpha * S[i]

           # # To plot decision boundary (only for 2D data)
           #  if((e+1)%1 == 0):
           #      dim_title = list(map(str,self.dims))
           #      dim_title = "_".join(dim_title)
           #      title = dim_title +"--"+ str(self.alpha)
           #      title += "--"+str(self.epoch_count)
           #      tplot = self.plot_decision_boundary(data, target)
           #      tplot.title(title)
           #      fname = "boundary/"+ title + "_" + str(e+1) + ".png"
           #      tplot.savefig(fname)
           
            loss = loss / len(data)
            lossList.append(loss)

        print("\nNetwork trained")
        plt.clf()
        plt.title(title)
        plt.plot(list(range(1,self.epoch_count+1)),lossList)
        plt.show()

    def predict(self,data):
        predList = list()
        for i in range(len(data)):
            sample = data[i]
            sample = sample.reshape((len(sample), 1))
            res = self.forwardPass(sample)
            predList.append(multilayerNN.indexOfMax(res[-1]))
        return np.array(predList)               