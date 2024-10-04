import errno

import tensorflow as tf
from tensorflow.keras import backend as K
import sklearn
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam
import os
import csv
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import random
import pandas as pd
import numpy as np
import math
import time
import flwr as fl
from decimal import *
import statistics
import seaborn as sns
import array
from scipy.stats import dirichlet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing

from imblearn.over_sampling import SMOTE
import sys

from Adult_utils import *
from fairness_utils import *




""""
attributes est un dictionnaire dont les clés sont les noms des colonnes sont les valeurs possibles
"""


#creer un model initialisé avec des poids nuls (ce model servira pour creer le modele global)
def Adult_NN_zero() :
    inputs = tf.keras.Input(shape=(88,))
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='zero', bias_initializer='zero')(inputs)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='zero', bias_initializer='zero')(x)
    x = tf.keras.layers.Dense(16, activation='relu', kernel_initializer='zero', bias_initializer='zero')(x)
    outputs = tf.keras.layers.Dense(1, activation='relu', kernel_initializer='zero', bias_initializer='zero')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.00001)

    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='recall')
    ]

    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model





def train(x, y, epch, client_id) :
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y)
    model = Adult_NN()
    history = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epch, verbose=2)
    #plot_learningCurve(history, epch, client_id) #Disable this blocking plot
    print ("evaluation sur les données de test ...")
    model.evaluate(x_test, y_test)
    return model


#Beta = 0 ---> FedAvg
def compute_weights1(fairness_values, beta) :
    for fairness_value in range(len(fairness_values)) :
        #Les valeurs negatives sont ramenés à 0
        if  fairness_values[i] >= math.sqrt(1/beta) or fairness_values[i] <= math.sqrt(1/beta) :
            fairness_values[i] = math.sqrt(1/beta)

    w = [math.exp(-beta * abs(value)) for value in fairness_values]
    normalization = sum(w)
    return [i/normalization for i in w]

def compute_weights2(fairness_values, beta) :
    w = [(-beta * (value**2)) + 1 for value in fairness_values]
    normalization = sum(w)
    return [i / normalization for i in w]

def compute_weights3(fairness_values, beta) :
    w = [(-beta * (value**2)) + 1 for value in fairness_values]
    return w


def scale_model_weights(weight, scalar):
    '''function for scaling a models weights'''
    if math.isnan(scalar) :
        scalar = 0.999
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar * weight[i])
    return weight_final


def sum_scaled_weights(scaled_weight_list):
    '''Return the sum of the listed scaled weights. The is equivalent to scaled avg of the weights'''
    avg_grad = list()
    # get the average grad accross all client gradients
    for grad_list_tuple in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tuple, axis=0)
        avg_grad.append(layer_mean)

    return avg_grad


def FedAvg(models, n, clients_weights, input_shape) :
    scaled_weights = []

    global_model = Adult_NN(input_shape)
    for i in range(n) :
        scaled_weights.append(scale_model_weights(models[i].get_weights(), clients_weights[i]))

    avg_weights = sum_scaled_weights(scaled_weights)

    global_model.set_weights(avg_weights)
    return global_model





#Retourne (P(Y=1 | S = 1), P(Y=1 | S = 0))
def positives_prop(data, val_1, val_0, attribute) :
    if attribute == 'Female' :
        attr_elements = data[data['sex'] == 1.0]
        non_attr_elements = data[data['sex'] == 0.0]
    else :
        attr_elements = data[data[attribute] == 1.0]
        non_attr_elements = data[data[attribute] == 0.0]
    positive_attr_elements = attr_elements[attr_elements['income']==1.0]
    positive_non_attr_elements = non_attr_elements[non_attr_elements['income'] == 1.0]
    return (((len(positive_attr_elements.axes[0])/len(data.axes[0]))/val_1),
    ((len(positive_non_attr_elements.axes[0]) / len(data.axes[0])) / val_0))

def compute_mkGlobal(model, d_client, attribute, val_1, val_0) :
    y_test = d_clients[i]['income']
    x_test = d_clients[i].drop('income', axis=1)

    pp_gr1 = compute_PP(x_test[x_test[attribute]==1.0], y_test[x_test[attribute]==1.0], model)
    pp_gr0 = compute_PP(x_test[x_test[attribute] == 0.0], y_test[x_test[attribute] == 0.0], model)

    (term_gr1, term_gr0) = positives_prop(d_client, val_1, val_0, attribute)

    return (pp_gr1 * term_gr1 - pp_gr0 * pp_gr0)
"""
def save_model(model, name, layers_indexes) :
    os.mkdir('../SharedFiles/Client_Models/Client'+str(name))
    for index in layers_indexes :
        weights = model.layers[index].get_weights()
        np.savetxt('../SharedFiles/Client_Models/Client'+str(name)+'/layer'+str(index)+'.csv', weights[0].flatten(), fmt='%s', delimiter=',')
"""



def update_local_model(agg_model, input_shape) :
    #update the local models from the aggregated one received from server
    local_model = tf.keras.models.clone_model(agg_model)
    local_model.build(input_shape)
    local_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='binary_crossentropy',
            metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Recall(name='Recall'),
            tf.keras.metrics.Precision(name='Precision')
                ]
            )

    local_model.set_weights(agg_model.get_weights())
    return local_model


def reconstruct_model(model_folder, normalization, input_shape, shapes=[(88, 16), (16, 16), (16, 1)]):
    # Get a list of all files in the model folder
    files = os.listdir(model_folder)
    # Filter files for kernel and bias files
    kernel_files = [f for f in files if 'kernel' in f]
    bias_files = [f for f in files if 'bias' in f]
    # Sort files to make sure they are in the correct order
    kernel_files.sort()
    bias_files.sort()
    weights = []
    # Iterate through the kernel and bias files to reconstruct each layer
    for shape, kernel_file, bias_file in zip(shapes, kernel_files, bias_files):
        # Extract layer index from the file name
        layer_index = int(kernel_file.split('_')[1])

        # Load weights and biases from CSV files
        kernel = []
        bias = []

        with open(os.path.join(model_folder, kernel_file), 'r', newline='') as csvfile:
           csvreader = csv.reader(csvfile)
           for line in csvreader:
               for value in line :
                    kernel.append(normalization * float(value))

        with open(os.path.join(model_folder, bias_file), 'r', newline='') as csvfile:
           csvreader = csv.reader(csvfile)
           for line in csvreader:
               for value in line :
                    bias.append(float(value))

        weights.append([np.array(kernel[:(shape[0] * shape[1])]).reshape(shape), np.array(bias[:shape[1]])])

    #model creation
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(16, activation='relu', weights=weights[0], name='dense')(inputs)
    x = tf.keras.layers.Dense(16, activation='relu', weights=weights[1], name='dense_1')(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', weights=weights[2], name='dense_2')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    optimizer = tf.keras.optimizers.Adamax(learning_rate=0.00001)
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='Precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=metrics
    )
    return model


def plot_error_density(approx_errors, filename) :

    sns.set_style("whitegrid")
    sns.histplot(approx_errors, kde=True)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xlabel('Values')
    plt.ylabel('Density')
    plt.ylim(0, 200)
    plt.xlim(-0.1, 0.1)
    #plt.title('$e_{agg}$ distribution Density ('+ str(len(approx_errors))+ ' samples)')
    plt.savefig('figs/error_density_plots/'+filename)
    plt.clf()
    plt.cla()
    plt.close()



set_global_determinism()


if __name__ == '__main__':
    data = pd.read_csv('datasets/adult.csv')
    data.replace('?', np.NaN)
    dataSet_summary(data)
    attributes = {
        'race'      : data['race'].unique(),
        'sex'       : data['sex'].unique(),
        'marital.status' : data['marital.status'].unique()
    }
    pre_processed_data = data_PreProcess(data)
    #Number of FL clients / iterations / epochs
    learning_iterations = 300
    n_clients = 3
    epochs = 1
    # Fairness Budget
    beta = 1.5
    aggregation_error_analysis = True
    #Definir l'attribut sur lequel l'analyse de l'équité sera faite
    attribute_to_manipulate = 'race'
    dataset_rows = len(data.axes[0])
    client_dataset_size = round(dataset_rows / n_clients)


    #Create a dictionnary with different heterogeneity levels (alphas)
    alphas = create_alphas(len (data[attribute_to_manipulate].unique()))
    d_clients = []
    #Diplay or not fairness plots
    display = False
    #Some data statistics
    (positive_prop_attr, positive_prop_non_attr) = positives_prop(pre_processed_data, 1, 1, 'White')
    for i in range(n_clients) :
        d_clients.append(Dirichelet_sampling(pre_processed_data, alphas['homogeneous'], data[attribute_to_manipulate].unique(), client_dataset_size))


    #Main loop
    for j in range(learning_iterations) :
        models = []
        metric_values = []
        mk_global_gr = []
        for i in range(n_clients) :
            #n_client = 1 --> Centralized learning.
            if n_clients == 1 :
                d_client = pre_processed_data

            y_client = d_clients[i]['income']
            x_client = d_clients[i].drop('income', axis=1)
            shape = (None, x_client.shape[1])

            if j == 0 :
                models.append(train_from_model(Adult_NN(shape), x_client, y_client, epochs, i+1))
            else :
                models.append(train_from_model(update_local_model(reconstructed_model, shape), x_client, y_client, epochs, i+1))

            save_model(models[i], str(i+1), j)
            #calcul de la métrique EOD pour les groupes White Black
            metric_values.append(EOD(models[i], x_client, y_client, 'White', 'Black'))
            save_metric(metric_values[i], str(i+1), j)
            #compute mk_global terms
            mk_global_gr.append(compute_mkGlobal(models[i], d_clients[i], 'White', positive_prop_attr, positive_prop_non_attr))
            print("mk_global = ", mk_global_gr)
            print("\n\n(Local) Fairness analysis : ")

        #Aggregate mk_global to get F_global
        F_global = sum([1/3 * i for i in mk_global_gr])
        F_global = 0.015
        #Les ecarts entre F_global et F_kt
        Deltas = [i - F_global for i in metric_values]

        print("F_global pour l'itération i = ", F_global)
        print("Deltas pour l'itération   i = ", Deltas)
        print("metric values = ", metric_values)

        w_metrics = compute_weights3(metric_values, beta)
        w_deltas = compute_weights3(Deltas, beta)
        print("weights from metrics = ", w_metrics)
        print("weights from Deltas = ", w_deltas)
        #Agggregate pour FairFed
        FairFed_global_model = FedAvg(models, n_clients, clients_weights=w_deltas, input_shape=shape)
        FairFed_global_model.save('models/model'+str(j))

        # Agggregate vanilla FedAvg
        FedAvg_global_model = FedAvg(models, n_clients, clients_weights=[1/n_clients for i in range(n_clients)], input_shape=shape)
        #tester le model global sur
        y = pre_processed_data['income']
        x = pre_processed_data.drop('income', axis=1)
        print("Global model [FedAvg] evaluation : ")
        FairFed_global_model.evaluate(x, y, verbose=2)
        print("Global model [FedAvg] evaluation : ")
        FedAvg_global_model.evaluate(x, y, verbose=2)
        #Fairness plots -> Turned off for now

        file_path = '../SharedFiles/AggModel/layer_0_kernel.csv'

        while not os.path.exists(file_path):
            time.sleep(1)
            print ("Waiting for LattigoModule to finish homomorphic aggregation ...")
        time.sleep(3)
        print('aggregation weights = ', w_deltas)
        normalization = 1/sum(w_deltas)
        reconstructed_model = reconstruct_model('../SharedFiles/AggModel', normalization, shape)
        agg_model_eval  = reconstructed_model.evaluate(x, y, verbose=0)
        print('\n\n\n\n evaluation of reconstructed model : ', agg_model_eval)

        #checking if the file to analyze exists
        if aggregation_error_analysis :
            apprximate_weights = []
            while not os.path.exists(file_path):
                time.sleep(1)
                print ("CKKS Variance Analysis Waiting for LattigoModule to finish homomorphic aggregation ...")
            #We need more float precision for a more accurate error estimation
            getcontext().prec += 10
            if os.path.isfile(file_path):
                with open(file_path, 'r', newline='') as csvfile:
                    csvreader = csv.reader(csvfile)
                    for line in csvreader:
                        for value in line :
                            apprximate_weights.append(float(value))

            else:
                raise ValueError("%s isn't a file!" % file_path)



            approx_errors = []
            print('\nsample of approximate model params from last layer : ')
            exact_layer_weights = FairFed_global_model.layers[1].get_weights()[0].flatten()
            print('apprximate values     Exact values')
            for i in range(10) :
                print(apprximate_weights[i], '        ', exact_layer_weights[i])
                approx_errors.append(exact_layer_weights[i] - apprximate_weights[i])

            print("\nObserving approximation errors : ")
            for i in range(10) :
                print(approx_errors[i])

            print("\napproximation erros mean : ")
            print(np.mean(np.array(approx_errors)))
            print("\napproximation erros stdev : ")
            print(statistics.stdev(approx_errors))
            print("\napproximation erros variance : ")
            print(statistics.stdev(approx_errors)**2)
