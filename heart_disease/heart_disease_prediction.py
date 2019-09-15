import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import normalize

print(tf.__version__)

if __name__ == '__main__':
    data = pd.read_excel('/home/elissonriller/comp_inteligence/heart_disease/dataexcel.xlsx')
    data = data.sample(frac=1)

    print(data)
    #print(data.head())
    #print(len(data.values))
    trainning_data = data.iloc[:-100] #Trainning data
    testing_data = data.iloc[194:] #Testing data

    #separating inputs and output
    input_trainning_data = trainning_data.iloc[:, range(13)] 

    output_trainning_data = trainning_data.iloc[:, [13]]

    input_testing_data = testing_data.iloc[:, range(13)]
    output_testing_data = testing_data.iloc[:, [13]]
    
    input_trainning_data = input_trainning_data.values
    output_trainning_data = output_trainning_data.values
    input_testing_data = input_testing_data.values
    output_testing_data = output_testing_data.values

    in_n_train = normalize(input_trainning_data)
    in_n_test = normalize(input_testing_data)



    output_trainning_data = np.array([[(1 if x == 0 else 0), (1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0), (1 if x == 4 else 0)] for x in output_trainning_data])
    output_testing_data = np.array([[(1 if x == 0 else 0), (1 if x == 1 else 0), (1 if x == 2 else 0), (1 if x == 3 else 0), (1 if x == 4 else 0)] for x in output_testing_data])
    
    print(len(output_trainning_data))

    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(13, input_shape=(len(in_n_train[0]),), activation=tf.nn.sigmoid))
    model.add(tf.keras.layers.Dense(5000, activation="tanh"))
    model.add(tf.keras.layers.Dense(5, activation="softmax"))

    lr = 0.001
    model.compile(loss='categorical_crossentropy',
                                 optimizer=tf.keras.optimizers.Adam(lr=lr),
                                 metrics=['accuracy', 'mse'])

    batch_size = 2
    #num_classes = 10
    epochs = 500

    history = model.fit(in_n_train, output_trainning_data,
                                       batch_size=batch_size,
                                       epochs=epochs,
                                       verbose=1,
                                       validation_data=(in_n_test, output_testing_data))

    loss_value, accuracy_value, mse_value = model.evaluate(in_n_test, output_testing_data)
    print("Loss value=", loss_value, "Accuracy value =", accuracy_value, "MSE value = ", mse_value)

    plt.plot(history.history['val_loss'], color='blue') #test
    plt.plot(history.history['loss'], color='red') #treinamento
    plt.show()

    # print(input_trainning_data)
    # print(output_trainning_data)

