import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
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

    #Obtener el numero de muestras que posee la clase minoritaria de los datos de entrenamiento
    num_muestras = []
    for i in range(len(clas)):
        r = np.where(pre_entrenamiento[0:, cdim-2] == clas[i])
        num_muestras.append(len(r[0]))

    clas_min = min(num_muestras)

    #Reduccion de las muestras
    entrenamiento = []
    for t in range(len(clas)):
        # Distancia Euclidiana enrte los datos de entrenamiento y su centroide
        for i in range(cren):
            suma = 0
            if pre_entrenamiento[i][cdim-2] == prot[t][cdim-2]:
                for j in range(cdim-2):
                    suma += math.pow(pre_entrenamiento[i][j] - prot[t][j], 2)
                dist = math.sqrt(suma)
                pre_entrenamiento[i][cdim-1] = dist
            
        #Ordenamiento de los datos de entrenamiento, por la distancia
        pre_entrenamiento = pre_entrenamiento[pre_entrenamiento[:,cdim-1].argsort()]

        #Determinacion de los datos de entrenamiento
        c = 0
        for i in range(cren):
            if pre_entrenamiento[i][cdim-2] == prot[t][cdim-2]:
                entrenamiento.append(pre_entrenamiento[i])
                c += 1
                if(c == clas_min):
                    break

        #Restablecer distancias a cero
        pre_entrenamiento[:,-1] = 0

    #Convertir entrenamiento a arreglo y liberar espacio
    entrenamiento = np.array(entrenamiento)
    pre_entrenamiento = []

    #Actualiza el numero de datos de entrenamiento
    cren = len(entrenamiento)

    #Volver a separar features de targets
    x_train = entrenamiento[:,0:-2]
    y_train = entrenamiento[:,-2]

    clf = SVC(kernel= 'linear', decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    pred = clf.predict(x_test)
    dec = clf.decision_function(x_test)
    #print dec
    #print y_test
    #print clf.predict(x_test)
    #print classification_report(y_test, clf.predict(x_test))
    #print clf.score(x_test, y_test)

    print geometric_mean_score(y_test, pred, average='macro')
    print f1_score(y_test, pred, average = "macro")
    print ""

    acc += geometric_mean_score(y_test, pred, average='macro')
    fscore += f1_score(y_test, pred, average = "macro")

print "/////////////////////////"
print acc/folds
print fscore/folds

"""
    pred = clf.predict(x)
    acc = clf.score(x, y)

    print pred
    print ""
    print acc

    print ""
    print clf.support_vectors_
"""
