import numpy as np
from scipy import io
import matplotlib.pyplot as plt
# **********************READ DATA**********************************************
def readData(trainFile, testFile):
    train = io.loadmat(trainFile)
    test = io.loadmat(testFile)
    ###TRAIN DATA###
    train_label = np.array(train['y'][0])
    train_vector = np.array(train['x']) / 255
    # train_vector = (train_vector -np.min(train_vector)) / (np.max(train_vector)-np.min(train_vector))
    # standardization
    # train_vector  = (train_vector - np.mean(train_vector)) / (10.0 * np.std(train_vector))

    ###TEST DATA###
    test_label = np.array(test['y'][0])
    test_vector = np.array(test['x']) / 255
    return train_label, train_vector, test_label, test_vector
#**************ONE HOT*********************************************************
def oneHot(label, class_size):
    y = label.shape[0]
    label_matrix = np.zeros([y, class_size], dtype=int)
    for i in range(y):
        correct_class = label[i]
        label_matrix[i][correct_class] = 1
    return label_matrix
#**********************SIGMOID FUNCTION****************************************
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
# ************************ReLu FUNCTION****************************************
def relu(x):
    return np.maximum(x, 0)

def derivative_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
#*************************TANH FUNCTION****************************************
def tanh(X):
    return np.tanh(X)
    
def derivative_tanh(x):
    return 1.0 - np.tanh(x) ** 2
# ***************SOFTMAX FUNCTION**********************************************
def softmax(out_array):
    for i in range(out_array.shape[0]):
        out_array[i] = np.exp(out_array[i]) / np.sum(np.exp(out_array[i]), axis=0)
    return out_array

def derivative_softmax(preds):
    return preds * (1 - preds)
#*****************LOSS FUNCTION************************************************
def loss(train_label, a1, size):
    result = -np.sum(np.multiply(train_label, np.log(a1)) + np.multiply(1 - train_label, np.log(1 - a1))) / size
    return result
# ******************************************************************************
# update parameters
def update_parameters_NN(parameters, grads, learning_rate=0.01):
    parameters = {"weight1": parameters["weight1"] - learning_rate * grads["dweight1"],
                  "bias1": parameters["bias1"] - learning_rate * grads["dbias1"],
                  "weight2": parameters["weight2"] - learning_rate * grads["dweight2"],
                  "bias2": parameters["bias2"] - learning_rate * grads["dbias2"]}
    return parameters
#################################################################################
# intialize parameters and layer sizes
def initialize_parameters_and_layer_sizes_NN(input_s, output_s, hidden_size):
    parameters = {"weight1": np.random.randn(input_s, hidden_size),
                  "bias1": 0.0,
                  "weight2": np.random.randn(hidden_size, output_s),
                  "bias2": 0.0}
    return parameters
#################################################################################
def forward_propagation_NN( train_vector, parameters):
    # this formula : oi = wijxj + bi
    Z1 = train_vector.dot(parameters["weight1"]) + parameters["bias1"]
    A1 = softmax(Z1/100)
    Z2 = A1.dot(parameters["weight2"]) + parameters["bias2"]
    A2 = softmax(Z2/100)

    cache = {"Z1": Z1,#h1
             "A1": A1,#a1
             "Z2": Z2,
             "A2": A2}

    return A2, cache
#################################################################################
# Compute cost
def compute_cost_NN(A2, Y, size):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/size
    return cost
#################################################################################
# Backward Propagation
def backward_propagation_NN(parameters, cache, X, Y,size):
    da2 = -((np.divide(Y,cache["A2"]) - np.divide(1-Y,1-cache["A2"])))/100
    dh2 = da2 * derivative_softmax(cache["A2"])
    dw2 = cache["A1"].T.dot(dh2)
    db2 = np.sum(dh2)
    dh1 = derivative_softmax(cache["Z1"]) * db2
    dw1 = np.matmul(X.T, dh1)
    db1 = np.sum(dh1)
    grads = {"dweight1": dw1,
             "dbias1": db1,
             "dweight2": dw2,
             "dbias2": db2}
    return grads
##################################################################################
def accuracy(x_test,y_test,parameters):
    preds = softmax((parameters["weight1"]) + parameters["bias1"] / 100)
    preds2 = softmax(preds.dot(parameters["weight2"]) + parameters["bias2"] / 100)
    count = 0
    for i, j in zip(y_test, np.argmax(preds2, axis=1)):
        if i == j:
            count += 1
    print("result accuracy")
    print(count / y_test.shape[0] * 100)
##################################################################################
def m_nn(x_train, y_train, x_test, y_test, epoch, learning_rate,batch_size,output_size, input_size , hidden_size):

    # initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(input_size,output_size,hidden_size)
    for i in range(epoch):
        # forward propagation
        size = x_train.shape[0]
        A2, cache = forward_propagation_NN(x_train, parameters)
        # compute cost
        compute_cost_NN(A2, y_train,size)
        # backward propagation
        grads = backward_propagation_NN(parameters,cache, x_train, y_train,size)
        # update parameters
        parameters = update_parameters_NN(parameters, grads)
        '''
        if i % 100 == 0:
            preds = softmax((parameters["weight1"]) + parameters["bias1"] / 100)
            preds2 = softmax(preds.dot(parameters["weight2"]) + parameters["bias2"] /100)
            count = 0
            for i, j in zip(y_test, np.argmax(preds2, axis=1)):
                if i == j:
                    count += 1

            print(count / y_test.shape[0] * 100)
        '''
    accuracy(x_test, y_test,parameters)

def main():
    # ********ASSIGN VALUE*******************************
    train_label, train_vector, test_label, test_vector = readData('train.mat', 'test.mat')
    epoch_size = 1000
    train_label = oneHot(train_label, 5)
    learning_rate = 0.01
    batch_size = 126
    output_size = 5
    input_size = 768
    hidden_size = 50
    # other typhical value 0.001, 0.01, 0.03, 0.1, 0.3, 1 etc.
    m_nn(train_vector, train_label, test_vector, test_label, epoch_size,learning_rate, batch_size, output_size, input_size , hidden_size)

if __name__ == '__main__':
    main()