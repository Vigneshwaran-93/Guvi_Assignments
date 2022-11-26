### Single Python File 

#Importing all necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
import seaborn as sns
df = pd.read_csv('data/train.csv')
df_test =pd.read_csv("data/test.csv")

#loading Datas
PHI = np.loadtxt('data/train.csv', dtype='float', delimiter=',', skiprows=1, usecols=tuple(range(1, 14)))
PHI_test = np.loadtxt('data/test.csv', dtype='float', delimiter=',', skiprows=1, usecols=tuple(range(1,14)))
Y = np.loadtxt('data/train.csv', dtype='float', delimiter=',', skiprows=1, usecols=14, ndmin=2)
PHI_test = np.concatenate((PHI_test, np.ones((105, 1))), axis=1)
PHI = np.concatenate((PHI, np.ones((400, 1))), axis=1)

#Scaling the data
from sklearn.preprocessing import MinMaxScaler
func = MinMaxScaler(feature_range=(0,1))
func.fit(PHI)
PHI = func.transform(PHI)
PHI_test = func.transform(PHI_test)
Y = np.log(Y)

#building the model
P_values = {'output_l2.csv': 2.0,
             'output_p1.csv': 1.75,
             'output_p2.csv': 1.5,
             'output_p3.csv': 1.3
             }

def error_change(p, PHI, w):
    if p == 2:
        deltaw = (2 * (np.dot(np.dot(np.transpose(PHI), PHI), w) - np.dot(np.transpose(PHI), Y)) + lambda1 * p * np.power(np.absolute(w), (p - 1)))
    if p < 2 and p > 1:
        deltaw = (2 * (np.dot(np.dot(np.transpose(PHI), PHI), w) - np.dot(np.transpose(PHI), Y)) + lambda1 * p * np.power(np.absolute(w), (p - 1)) * np.sign(w))
    return deltaw


for (fname, p) in P_values.items():
    # Set w to 0's
    w = np.zeros((14, 1))
    lambda1 = 0.2 #hyperparameter 
    t = 0.00012 #step size

    w_new = w - t * error_change(p, PHI , w)

    i = 0
    
    while(np.linalg.norm(w_new-w) > 10 ** -10):
        w = w_new
        w_new = w - t * error_change(p, PHI, w)
        i = i + 1
  
    id_test = np.loadtxt('data/test.csv', dtype='int', delimiter=',', skiprows=1, usecols=0, ndmin=2)

    # predecting y for test data
    y_test = np.exp(np.dot(PHI_test, w_new))
    df_test[fname] = y_test
    # Save the ids and y
    np.savetxt(fname, np.concatenate((id_test, y_test), axis=1),
               delimiter=',', fmt=['%d', '%f'], header='ID,MEDV', comments='')

print(df_test.head())