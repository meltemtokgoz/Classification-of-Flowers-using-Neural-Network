import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import argparse
import os
#***********************PARAMETERS*********************************************
def argumentParser():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("-td", "--traindata", default='train.mat')
    ap.add_argument("-ttd", "--testdata", default='test.mat')
    ap.add_argument("-i", "--inputsize", default='768')
    ap.add_argument("-o", "--outputsize", default='5')
    ap.add_argument("-b", "--batchsize", default='128')
    ap.add_argument("-e", "--epoch", default='3000')
    ap.add_argument("-lr", "--learningrate", default='0.01')
    return vars(ap.parse_args())
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
def initialize_parameters(input_s,output_s):
    initial_b = 0.0
    initial_w = np.random.randn(input_s, output_s)
    return initial_w, initial_b
#******************************************************************************
def feedforward(train_vector, initial_w, initial_b):
    # this formula : oi = wijxj + bi
    o1 = train_vector.dot(initial_w) + initial_b
    # send to activation function:
    a1 = softmax(o1 / 100)
    return a1
# ******************************************************************************
def backpropagation(train_label, train_vector, a1, initial_w, initial_b, learning_rate):
    # Backpropagation
    da1 = -(np.divide(train_label, a1) - np.divide(1 - train_label, 1 - a1)) / 100
    dh1 = da1 * derivative_softmax(a1)
    new_weight = train_vector.T.dot(dh1)
    new_bias = np.sum(dh1)

    # update weight and bias
    initial_w -= learning_rate * new_weight
    initial_b -= learning_rate * new_bias
    return initial_w, initial_b
# *****************************************************************************
def accuracy(test_vector,test_label,initial_w,initial_b):
    preds = softmax((test_vector.dot(initial_w) + initial_b) / 100)
    count = 0
    for i, j in zip(test_label, np.argmax(preds, axis=1)):
        if i == j:
            count += 1
    print("Accuracy : ")
    print(count / test_vector.shape[0] * 100)
# *****************************************************************************
def s_nn(train_vector, train_label, test_vector, test_label, batch_size, initial_w, initial_b, learning_rate,epoch_size):
    count = 0
    for epoch in range(epoch_size):
        g = []
        for i in range(0, len(train_vector), batch_size):
            size = train_vector[i:i + batch_size].shape[0]
            a1 = feedforward(train_vector[i:i + batch_size], initial_w, initial_b)
            # send to loss function:
            loss_val = loss(train_label[i:i + batch_size], a1, size)
            g.append(loss_val)
            intial_w, initial_b = backpropagation(train_label[i:i + batch_size], train_vector[i:i + batch_size]
                                                  , a1, initial_w, initial_b, learning_rate)
            count += 1
        '''
        if epoch % 100 == 0:
            preds = softmax((test_vector.dot(initial_w) + initial_b) / 100)
            count = 0
            for i, j in zip(test_label, np.argmax(preds, axis=1)):
                if i == j:
                    count += 1

            print(count / test_vector.shape[0] * 100)
        '''
    accuracy(test_vector, test_label, initial_w, initial_b)

    g.clear
    plt.plot(g)
    plt.show()
# ******************************************************************************    
def main():
    args = argumentParser()
    train_label, train_vector, test_label, test_vector = readData(args['traindata'], args['testdata'])
    train_label = oneHot(train_label, int(args['outputsize']))
    initial_w , initial_b = initialize_parameters(int(args['inputsize']),int(args['outputsize']))         
    # other typhical value 0.001, 0.01, 0.03, 0.1, 0.3, 1 etc.
    s_nn(train_vector, train_label, test_vector, test_label, int(args['batchsize']), initial_w, initial_b,float(args['learningrate']), int(args['epoch']))
    # save learned parameters
    if not os.path.isdir("../model"):
        os.mkdir("../model")
    np.savez("../model/model.npz", weights=initial_w, bias= initial_b )
    print("Parameters are saved..")
    # ******************************************************************************

if __name__ == '__main__':
    main()
    
