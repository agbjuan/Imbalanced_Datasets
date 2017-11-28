import numpy as np
import math
import pandas
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold
from sklearn import preprocessing
from imblearn.metrics import geometric_mean_score

#Cargar base de datos
datos = np.loadtxt("pageblocks.dat", delimiter = ",")
np.random.shuffle(datos)
features = datos[:,[0,4,3,9,6,5]]
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
    zeros = np.zeros((len(entrenamiento), 2))
    entrenamiento = np.concatenate((entrenamiento, zeros), axis = 1)
    pruebas = x_test
    zeros = np.zeros((len(pruebas), 1))
    for i in zeros:
        i += 1
    pruebas = np.concatenate((pruebas, zeros), axis = 1)
    #####

    cren = len(entrenamiento)
    cdim = len(entrenamiento[0])
    cren2 = len(pruebas)

    #Calcular masa de los datos de entrenamiento
    #Se guarda una lista de las clases de los datos de entrenamiento
    clas = []
    for i in range(cren):
        if entrenamiento[i][cdim-3] not in clas:
            clas.append([entrenamiento[i][cdim-3]])
            
    suma = 0
    #Contar cuantos elementos existen de cada clase
    for i in range(len(clas)):
        r = np.where(entrenamiento[0:,cdim-3] == clas[i][0])
        clas[i].append(len(r[0]))
        suma += len(r[0])

    #Determinar el porcentaje que corresponde a cada clase de los elementos totales
    for i in range(len(clas)):
        clas[i][1] = float(clas[i][1]) / float(suma)
    
    #Asignar masa a los objetos
    for i in range(cren):
        for j in range(len(clas)):
            if entrenamiento[i][cdim-3] == clas[j][0]:
                entrenamiento[i][cdim-2] = 1/math.pow(clas[j][1], 2)

    for k in range(1, cren + 1):
        resultados = []
        for t in range(cren2):

            #print pruebas[t]
            #print ""
            # Distancia Euclidiana enrte el objeto a clasificar y los datos de train
            for i in range(cren):
                suma = 0
                for j in range(cdim-3):
                    suma += math.pow(entrenamiento[i][j] - pruebas[t][j], 2)
                dist = math.sqrt(suma)
                entrenamiento[i][cdim-1] = dist

            #print entrenamiento

            # Odenamiento de los vecinos, por la distancia
            entrenamiento = entrenamiento[entrenamiento[:,cdim-1].argsort()]

            #print entrenamiento
            #print ""

            # Obtener las diferentes clases que existen entre los vecinos mas cercanos
            clasificaciones = []
            for i in range(k):
                if entrenamiento[i][cdim-3] not in clasificaciones:
                    clasificaciones.append(entrenamiento[i][cdim-3])

            #print clasificaciones
            #print ""
            
            # Calculo de las fuerzas totales entre cada clase y el objeto a clasificar
            fuerzas = []
            for i in range(len(clasificaciones)):
                f = 0
                for j in range(k):
                    if clasificaciones[i] == entrenamiento[j][cdim-3]:
                        if entrenamiento[j][cdim-1] != 0:
                            f += entrenamiento[j][cdim-2]*pruebas[t][cdim-3]/math.pow(entrenamiento[j][cdim-1], 2)
                fuerzas.append(f)

            #print fuerzas
            
            # Prediccion de la clase
            c = clasificaciones[fuerzas.index(max(fuerzas))]
            
            #print ""
            #print c
            resultados.append(c)
            #print "////////////////////////////////////////////////////"

        #print resultados
        #print y_test

        if k == 1:
            acc_best = geometric_mean_score(y_test, resultados, average='macro')
            k_best = k
            fscore = f1_score(y_test, resultados, average = "macro")
        else:
            acc = geometric_mean_score(y_test, resultados, average='macro')
            if acc > acc_best:
                acc_max = acc
                k_best = k
                fscore = f1_score(y_test, resultados, average = "macro")

    print "Vecinos considerados: ", k_best
    print "G-mean: ", acc_best
    print "F-Score: ", fscore
    print "------------------------"
    acc_sum += acc_best
    fscore_sum += fscore

print ""
print "///////////////////////"
print "G-mean (avr): ", acc_sum/folds
print "F-Score (avr): ", fscore_sum/folds
    
"""
print matriz_c
print ""
print accuracy_score(resultados, y_test)
print ""
print classification_report(y_test, resultados)
print ""
print f1_score(resultados, y_test, average = "macro")
"""
