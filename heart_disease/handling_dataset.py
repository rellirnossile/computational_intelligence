import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

if __name__ == '__main__':
    data = pd.read_csv('~/comp_inteligence/heart_disease/switzerland.data', header=None, sep='^')
    #print(len(data))
    test = []
    for i in range(len(data)):
        test.append(data.iloc[i].tolist())
    s = []
    for i in test:  
        for j in i[0].split(' '):
            s.append(j)
    a = np.array(s)
    b = np.reshape(a, (123,76))
    df = pd.DataFrame(data=b, index=range(123), columns=range(76))
    #print(df.head())
    df.to_csv('~/comp_inteligence/heart_disease/dataset.data')