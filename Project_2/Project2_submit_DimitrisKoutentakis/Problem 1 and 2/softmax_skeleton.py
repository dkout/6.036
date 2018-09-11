import sys
sys.path.append("..")
import utils
from utils import *
import numpy as np
import matplotlib.pyplot as plt

def augmentFeatureVector(X):
    columnOfOnes = np.zeros([len(X), 1]) + 1
    return np.hstack((columnOfOnes, X))

def computeProbabilities(X, theta, tempParameter):
    #YOUR CODE HERE
    H = np.dot(theta, np.transpose(X))
    H = np.divide(H, tempParameter)
    c = np.amax(H, axis=0)
    H = np.exp(H-c)
    Hsum = np.sum(H, axis=0)
    H = np.divide(H, Hsum)
    return H
    

def computeCostFunction(X, Y, theta, lambdaFactor, tempParameter):


    reg = lambdaFactor*0.5*np.sum(np.square(theta))  #regularization
    logH = np.log(computeProbabilities(X, theta, tempParameter))
    n = X.shape[0]
    k = X.shape[1]
    cost = 0
    for i in range(n):
        cost-=logH[Y[i],i]
        # for j in range(k):
            # cost -= logH[j,i] if Y[i]==j else 0
    cost = cost/n +reg
    return cost



def runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter):

    thetaprime = []
    n = X.shape[0]
    k = theta.shape[0]
    d=theta.shape[1]
    #print (theta[:,0])
    H = computeProbabilities(X, theta, tempParameter)
    der=np.zeros([k,d])
    # print (theta.shape)
    # print (Y.shape)
    label = (np.matrix(Y))==np.transpose(np.matrix(np.arange(10)))
    # print (label)
    label = label- H
    # print(label)

    der -= np.dot(label,X)/(n*tempParameter)
    # print (der)
    der = np.add(der, theta*lambdaFactor)
    thetaprime = np.add(theta, -alpha*der)
    # print (thetaprime.shape)
    return thetaprime

def updateY(trainY, testY):
    return (trainY%3, testY%3)

def getClassification(X, theta, tempParameter):
    X = augmentFeatureVector(X)
    probabilities = computeProbabilities(X, theta, tempParameter)
    return np.argmax(probabilities, axis = 0)
    

def computeTestErrorMod3(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)%3
    return 1 - np.mean(assignedLabels == Y)
    

def softmaxRegression(X, Y, tempParameter, alpha, lambdaFactor, k, numIterations):
    X = augmentFeatureVector(X)
    theta = np.zeros([k, X.shape[1]])
    costFunctionProgression = []
    for i in range(numIterations):
        costFunctionProgression.append(computeCostFunction(X, Y, theta, lambdaFactor, tempParameter))
        theta = runGradientDescentIteration(X, Y, theta, alpha, lambdaFactor, tempParameter)
    return theta, costFunctionProgression
    


def plotCostFunctionOverTime(costFunctionHistory):
    plt.plot(range(len(costFunctionHistory)), costFunctionHistory)
    plt.ylabel('Cost Function')
    plt.xlabel('Iteration number')
    plt.show()

def computeTestError(X, Y, theta, tempParameter):
    errorCount = 0.
    assignedLabels = getClassification(X, theta, tempParameter)
    return 1 - np.mean(assignedLabels == Y)
