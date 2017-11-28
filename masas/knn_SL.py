import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from imblearn.metrics import geometric_mean_score

#Cargar base de datos
datos = np.loadtxt("new-thyroid.dat", delimiter = ",")
np.random.shuffle(datos)
features = datos[:,[1,4,0,2]]
targets = datos[:,-1]

#Reescalar entre 0-1
#min_max_scaler = preprocessing.MinMaxScaler()
#features = min_max_scaler.fit_transform(features)

folds = 5
kf = KFold(n_splits = folds)

acc_sum = 0
fscore_sum = 0
for train, test in kf.split(features):
    x_train, x_test = features[train], features[test]
    y_train, y_test = targets[train], targets[test]
    
    y_train = y_train.tolist()
    y_train = np.array([y_train]).T

    #####
    entrenamiento = np.concatenate((x_train, y_train), axis = 1)
    zeros = np.zeros((len(entrenamiento), 1))
    entrenamiento = np.concatenate((entrenamiento, zeros), axis = 1)
    pruebas = x_test
    #####

    #Dimensiones de la base de datos
    cren = len(entrenamiento)
    cdim = len(entrenamiento[0])
    cren2 = len(pruebas)
    
    for k in range(1, len(x_train) + 1):
        neigh = KNeighborsClassifier(n_neighbors = k, weights = 'uniform')
        #weights = 'distance' para kNN ponderado
        neigh.fit(x_train,y_train)

        #a = y_test.values
        pred = neigh.predict(x_test)

        matriz_c = confusion_matrix(y_test, pred, labels = None, sample_weight = None)

        #print accuracy_score(y_test, pred)
        #print f1_score(y_test, pred, average = "macro")
        #print ""

        if k == 1:
            acc_best = geometric_mean_score(y_test, pred, average='macro')
            k_best = k
            fscore = f1_score(y_test, pred, average = "macro")
        else:
            acc = geometric_mean_score(y_test, pred, average='macro')
            if acc > acc_best:
                acc_max = acc
                k_best = k
                fscore = f1_score(y_test, pred, average = "macro")

    print "Vecinos considerados: ", k_best
    print "Accuracy: ", acc_best
    print "F-Score: ", fscore
    print "------------------------"
    acc_sum += acc_best
    fscore_sum += fscore

print ""
print "///////////////////////"
print "G-mean (avr): ", acc_sum/folds
print "F-Score (avr): ", fscore_sum/folds

"""
    print y_test
    print matriz_c
    print ""
    print neigh.score(x_test, y_test)
    print ""
    print classification_report(a, pred)
"""





