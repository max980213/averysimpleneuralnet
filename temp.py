# a very simple version DNN

# import tensorflow as tf
import numpy as np
import scipy.special
import matplotlib.pyplot as plt

class NN:
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.wih = np.random.normal(0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0, pow(self.inodes, -0.5), (self.onodes, self.hnodes))
        # w-in-hidden and w-hidden-out
        self.lr = learningrate
        self.act_f = lambda x: scipy.special.expit(x)
        # 激活函数
        pass
    
    def train(self, inputs_list, targets_list):  # 训练神经网络
        targets = np.array(targets_list, ndmin = 2).T
        inputs, hidden_outputs, final_outputs = self.in_out(inputs_list)
        outputs_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, outputs_errors)  # 得到隐藏层的误差
        self.who += self.lr * np.dot((outputs_errors * final_outputs * (1.0 - final_outputs)), hidden_outputs.T)
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), inputs.T)
        # 对最终输出对ωi,j求偏导，只剩下系数，即上一层的输出，这一层的raw输入，没乘权重的
        # numpy矩阵可以直接与实数进行运算，每个元素都进行同样的运算
        # * 是矩阵与矩阵之间对应元素的乘法
        pass
    
    def in_out(self, input_list): # 一次完整输入输出的流程
        inputs = np.array(input_list, ndmin = 2).T  # T是转置，ndmin是指定数组的最小维度
        # numpy的数组有空维的情况，不可转置。行列向量，ndmin需要设置为2
        hidden_inputs = np.dot(self.wih, inputs)  # 矩阵乘法
        hidden_outputs = self.act_f(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.act_f(final_inputs)
        return inputs, hidden_outputs, final_outputs
        pass
    pass

input_nodes = 28*28
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.3

n = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

training_datafile = open('C:\\Users\\HaoRan Zhu\\Desktop\\mnist_train.csv','r')
record = training_datafile.readline()

while record:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    targets = np.zeros(output_nodes) + 0.01
    targets[int(all_values[0])] = 0.99
    n.train(inputs, targets)
    record = training_datafile.readline()
    pass

training_datafile.close()

test_file = open('C:\\Users\\HaoRan Zhu\\Desktop\\mnist_test_10.csv','r')
test_data = test_file.readlines()
test_file.close()

for record in test_data:
    all_values = record.split(',')
    inputs = np.asfarray(all_values[1:]) / 255 * 0.99 + 0.01
    inputs, hidden_outputs, final_outputs = n.in_out(inputs)
    maxplc = np.argmax(final_outputs)
    print(all_values[0], maxplc)
    pass
# 浅度学习之神经网络    
    
    
    
    
    
    