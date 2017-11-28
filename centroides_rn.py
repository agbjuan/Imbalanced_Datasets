import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from imblearn.metrics import geometric_mean_score

def centroides(cdim, cren, clas, entrenamiento, prot):
    for i in range(len(clas)):
        c = 0
        for r in range(cren):
            if clas[i] == entrenamiento[r][cdim-2]:
                c += 1
                for j in range(cdim-2):
                    prot[i][j] += entrenamiento[r][j]
        for j in range(cdim-2):
            prot[i][j] /= c

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

acc = 0
fscore = 0
for train, test in kf.split(features):
    x_train, x_test = features[train], features[test]
    y_train, y_test = targets[train], targets[test]
    
    y_train = y_train.tolist()
    y_train = np.array([y_train]).T

    #####
    pre_entrenamiento = np.concatenate((x_train, y_train), axis = 1)
    zeros = np.zeros((len(pre_entrenamiento), 1))
    pre_entrenamiento = np.concatenate((pre_entrenamiento, zeros), axis = 1)
    pruebas = x_test
    #####

    #Dimensiones de la base de datos
    cren = len(pre_entrenamiento)
    cdim = len(pre_entrenamiento[0])
    cren2 = len(pruebas)
    
    #Se guarda una lista de las clases de los datos de entrenamiento
    clas = []
    for i in range(cren):
        if pre_entrenamiento[i][cdim-2] not in clas:
            clas.append(pre_entrenamiento[i][cdim-2])

    #Se generan los centroides de los datos de entrenamiento
    prot = np.zeros((len(clas), cdim))
    for i in range(len(clas)):
        prot[i][cdim-2] = clas[i]

    centroides(cdim, cren, clas, pre_entrenamiento, prot)

    #Liberar espacio
    pre_entrenamiento = []

    #Actualiza el numero de datos de entrenamiento
    cren = len(prot)
    
    #Volver a separar features de targets
    x_train = prot[:,0:-2]
    y_train = prot[:,-2]
    
    rn = MLPClassifier(max_iter = 1000, hidden_layer_sizes=(10),
                       activation = 'logistic', solver = 'lbfgs',
                       learning_rate = 'constant', learning_rate_init = 0.2)

    rn.fit(x_train,y_train)

    tar = y_test
    pred = rn.predict(x_test)
    matriz_c = confusion_matrix(tar, pred, labels = None, sample_weight = None)

    print geometric_mean_score(y_test, pred, average='macro')
    print f1_score(y_test, pred, average = "macro")
    print ""

    acc += geometric_mean_score(y_test, pred, average='macro')
    fscore += f1_score(y_test, pred, average = "macro")

print "/////////////////////////"
print acc/folds
print fscore/folds
    
"""
    print matriz_c
    print ""
    print classification_report(tar, pred)
    print ""
    print rn.score(x_test, y_test)
"""
