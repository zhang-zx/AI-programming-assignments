# classifiers.py
# -------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util
from classificationMethod import ClassificationMethod
import numpy as np


class LinearRegressionClassifier(ClassificationMethod):
    """
    Classifier with Linear Regression.
    """
    def __init__(self, legalLabels):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(LinearRegressionClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'lr'
        self.lambda_ = 1e-4
        self.weights = None

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        For doc data, trainingData/validationData should also be in numpy format.
        """
        n, dim = trainingData.shape
        X = trainingData
        Y = np.zeros((n, len(self.legalLabels)))
        Y[np.arange(n), trainingLabels] = 1
        self.weights = np.dot(np.linalg.inv(np.dot(X.T, X) + self.lambda_*np.eye(dim)), np.dot(X.T, Y))
    
    def classify(self, data):
        """
        Predict which class is in.
        :param data: data to classify which class is in. (in numpy format)
        :return list or numpy array
        """
        return np.argmax(np.dot(data, self.weights), axis=1)


class KNNClassifier(ClassificationMethod):
    """
    KNN Classifier.
    """
    
    def __init__(self, legalLabels, num_neighbors):
        """

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param num_neighbors: number of nearest neighbors.
        """
        super(KNNClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = 'knn'
        self.num_neighbors = num_neighbors
    
    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        """
        Train the Linear Regression Classifier by just storing the trainingData and trainingLabels.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        """

        # trainingData is normalized
        self.trainingData = trainingData / np.linalg.norm(trainingData, axis=1).reshape((len(trainingData), 1))
        self.trainingLabels = trainingLabels

    def sim(self, X):
        num_test = X.shape[0]
        num_train = self.trainingData.shape[0]
        dists = np.zeros((num_test, num_train))
        test_sum=np.sum(np.square(X),axis=1)
        train_sum=np.sum(np.square(self.trainingData),axis=1)
        inner_product=np.dot(X,self.trainingData.T)
        dists=np.sqrt(-2*inner_product+test_sum.reshape(-1,1)+train_sum)
        return dists
    
    def classify(self, data):
        """
        Predict which class is in.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.sort, np.argsort: return a sorted copy (or indices) of an array.

        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """

        data = data / np.linalg.norm(data, axis=1).reshape((len(data), 1))

        "*** YOUR CODE HERE ***"
        # should compute sim(data[i], data[j]) = dot(data[i], data[j])
        k = self.num_neighbors
        dists = self.sim(data)
        num_test=dists.shape[0]
        y_pred=np.zeros(num_test)
        for i in range(num_test):
            closest_y=[]
            y_indicies=np.argsort(dists[i,:],axis=0) 
            closest_y=self.trainingLabels[y_indicies[: k]] 
            y_pred[i]=np.argmax(np.bincount(closest_y))
        return y_pred


class PerceptronClassifier(ClassificationMethod):
    """
    Perceptron classifier.
    """
    def __init__( self, legalLabels, max_iterations):
        """
        self.weights/self.bias: parameters to train, can be considered as parameter W and b in a perception.
        self.batchSize: batch size in a mini-batch, used in SGD method
        self.weight_decay: weight decay parameters.
        self.learningRate: learning rate parameters.

        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        :param max_iterations: maximum epoches
        """
        super(PerceptronClassifier, self).__init__(legalLabels)
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1
        
    def setWeights(self, input_dim):
        self.weights = np.random.randn(input_dim, len(self.legalLabels))/np.sqrt(input_dim)
        self.bias = np.zeros(len(self.legalLabels))
    
    def prepareDataBatches(self, traindata, trainlabel):
        """
        Generate data batches with given batch size(self.batchsize)

        :return a list in which each element are in format (batch_data, batch_label). E.g.:
            [(batch_data_1), (batch_label_1), (batch_data_2, batch_label_2), ..., (batch_data_n, batch_label_n)]

        """
        index = np.random.permutation(len(traindata))
        traindata = traindata[index]
        trainlabel = trainlabel[index]
        split_no = int(len(traindata) / self.batchSize)
        return zip(np.split(traindata[:split_no*self.batchSize], split_no), np.split(trainlabel[:split_no*self.batchSize], split_no))

    def train(self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        The training loop for the perceptron passes through the training data several
        times and updates the weight vector for each label based on classification errors.
        See the project description for details.

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)

        Some data structures that may be in use:
        self.weights/self.bias (numpy format): parameters to train,
            can be considered as parameter W and b in a perception.
        self.batchSize (scalar): batch size in a mini-batch, used in SGD method
        self.weight_decay (scalar): weight decay parameters.
        self.learningRate (scalar): learning rate parameters.

        Some numpy functions that may be of use (we consider np as short of numpy)
        np.sum(a, axis): sum of array elements over a given axis.
        np.dot(A, B): dot product of two arrays, or matrix multiplication between A and B.
        np.mean(a, axis): mean value of array elements over a given axis
        np.exp(a)
        """

        self.setWeights(trainingData.shape[1])
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
        
        # Hyper-parameters. Your can reset them. Default batchSize = 100, weight_decay = 1e-3, learningRate = 1
        "*** YOU CODE HERE ***"
        self.batchSize = 100
        self.weight_decay = 1e-3
        self.learningRate = 1

        for iteration in range(self.max_iterations):
            print "Starting iteration ", iteration, "..."
            dataBatches = self.prepareDataBatches(trainingData, trainingLabels)
            for batchData, batchLabel in dataBatches:
                "*** YOUR CODE HERE ***"
                for i in range(len(batchData)):
                    top_score = -1
                    best_label = None
                    current_data = batchData[i] 
                    for label in self.legalLabels:
                        result = np.dot((self.weights.T)[label] , current_data )
                        if result > top_score:
                            top_score = result
                            best_label = label
                    actual_label = batchLabel[i]
                    if best_label != actual_label:  
                        (self.weights.T)[actual_label] = ((self.weights.T)[actual_label] + current_data) 
                        (self.weights.T)[best_label] = (self.weights.T)[best_label] - current_data.T 

    def classify(self, data):
        """
        :param data: Data to classify which class is in. (in numpy format)
        :return Determine the class of the given data (list or numpy array)
        """
        
        return np.argmax(np.dot(data, self.weights) + self.bias, axis=1)

    def visualize(self):
        sort_weights = np.sort(self.weights, axis=0)
        _min = sort_weights[524]
        _max = sort_weights[-10]
        return np.clip(((self.weights-_min) / (_max-_min)).T, 0, 1)


class SVMClassifier(ClassificationMethod):
    """
    SVM Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(SVMClassifier, self).__init__(legalLabels)
        self.type = 'svm'
        self.legalLabels = legalLabels
        
        # you may use this for constructing the svm classifier with sklearn
        self.sklearn_svm = None 
        
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        training with SVM using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)

        sklearn.svm.SVC should be used in this algorithm. The following parameters should be taken into account:
        C: float
        kernel: string
        gamma: float
        decision_function_shape:  'ovo' or 'ovr'
        """
        from sklearn import svm
         
        # "*** YOUR CODE HERE ***"
        self.sklearn_svm = svm.SVC(C = 10, gamma=0.015, kernel='rbf', probability=True)
        self.sklearn_svm.fit(trainingData, trainingLabels)

    
        # util.raiseNotDefined()
        
    
    def classify(self, data):
        """
        classification with SVM using sklearn API
        """
        "*** YOUR CODE HERE ***"
        return self.sklearn_svm.predict(data)
        # util.raiseNotDefined()

        
class BestClassifier(ClassificationMethod):
    """
    Best Classifier
    """
    def __init__(self, legalLabels):
        """
        :param legalLabels: Labels to predict (for digit data, legalLabels = range(10))
        """
        super(BestClassifier, self).__init__(legalLabels)
        self.type = 'best'
        self.legalLabels = legalLabels
        
        "*** YOUR CODE HERE (If needed) ***"
        self.sklearn_svm = None  
        self.compontNum = 35
        self.pca = None

    
    def train( self, trainingData, trainingLabels, validationData, validationLabels ):
        """
        design a classifier using sklearn API

        For digit data, trainingData/validationData are all in numpy format with size ([number of data], 784)
        """
        "*** YOUR CODE HERE ***"
        from sklearn.decomposition import PCA
        from sklearn.svm import SVC
        self.pca = PCA(n_components=self.compontNum, whiten=True)
        self.pca.fit(trainingData) 
        trainingData = np.vstack((trainingData, validationData))
        trainingLabels = np.hstack((trainingLabels, validationLabels))
        trainingData = self.pca.transform(trainingData)
        self.sklearn_svm = SVC(C=3, gamma=1/35.0)
        self.sklearn_svm.fit(trainingData, trainingLabels)

    
    
    def classify(self, data):
        """
        classification with the designed classifier
        """
        "*** YOUR CODE HERE ***"
        data = self.pca.transform(data)
        return self.sklearn_svm.predict(data)

