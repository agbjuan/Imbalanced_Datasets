import numpy as np
import math
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn import preprocessing
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

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

    #Se guarda una lista de las clases de los datos de entrenamiento
    clas = []
    for i in range(cren):
        if entrenamiento[i][cdim-2] not in clas:
            clas.append(entrenamiento[i][cdim-2])

    #Se generan los centroides de los datos de entrenamiento
    prot = np.zeros((len(clas), cdim))
    for i in range(len(clas)):
        prot[i][cdim-2] = clas[i]

    centroides(cdim, cren, clas, entrenamiento, prot)

    resultados = []
    for t in range(cren2):

        #print pruebas[t]
        #print ""
        # Distancia Euclidiana entre el objeto a clasificar y los centroides
        for i in range(len(prot)):
            suma = 0
            for j in range(cdim-2):
                suma += math.pow(prot[i][j] - pruebas[t][j], 2)
            dist = math.sqrt(suma)
            prot[i][cdim-1] = dist

        # Ordenamiento de los centroides, por la distancia
        prot = prot[prot[:,cdim-1].argsort()]
            
        # Prediccion de la clase
        c = prot[0][cdim-2]
            
        resultados.append(c)

    #print resultados
    #print y_test

    fscore = f1_score(y_test, resultados, average = "macro")
    print "G-mean: ", acc
    print "F-Score: ", fscore
    print "------------------------"
    acc_sum += acc
    fscore_sum += fscore

print ""
print "///////////////////////"
print "G-mean (avr): ", acc_sum/folds
print "F-Score (avr): ", fscore_sum/folds
