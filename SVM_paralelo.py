import itertools
import multiprocess
import time
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# Define the parameter grid for SVM
param_grid_svm = {
    'C': [0.2, 1, 9],  #El margen de tolerancia de la clasificación
    'kernel':['linear', 'rbf'], # 'poly', 'sigmoid' Se excluyen para acelerar el proceso de evaluación
    'gamma': [0.001, 0.01, 0.1]  #Se usa para el kernel rbf 
}

#Funcion para calcular la precision, recall y f1
def analyze_precision_recall_f1(y_test, y_pred):  
    recall = recall_score(y_test, y_pred, average='weighted') #Calcula el recall 
    f1 = f1_score(y_test, y_pred, average='weighted') #Calcula el f1
    return recall, f1

#Funcion para evaluar SVM para un conjunto de hiperparametros
def evaluate_svm(hyperparameter_set, X_train, y_train, X_test, y_test, lock,output_file):
    for params in hyperparameter_set:
        start_time = time.perf_counter()  #tiempo de inicio
        
        # Se crea el modelo de SVM con los parametros dados 
        # Se excluyen los kernels polinomico y sigmoidal para acelerar el proceso de evaluación 
        if params['kernel'] == 'rbf': #Se evalua si el kernel es rbf para incluir el parametro gamma
            model = SVC(C=params['C'], kernel=params['kernel'], gamma=params['gamma'])
        else: #Si no es rbf, se excluye el parametro gamma
            model = SVC(C=params['C'], kernel=params['kernel'])
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        end_time = time.perf_counter()#Tiempo de fin 
        
        accuracy = accuracy_score(y_test, y_pred)
        recall, f1 = analyze_precision_recall_f1(y_test, y_pred)#Calcular precision, recall y f1
        time_taken = end_time - start_time#Tiempo total
        #Aqui se imprime el resultado de la evaluación 
        lock.acquire()
        with open(output_file, "a") as f:
            f.write(f"Parametros: {params}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Tiempo: {time_taken:.4f} segundos\n")
        lock.release()

# Funcion para nivelar la carga de trabajo entre los procesos, Obtenida de codigo compartido en Classroom
def nivelacion_cargas(D, n_p):
    s = len(D) % n_p
    n_D = D[:s]
    t = int((len(D) - s) / n_p)
    out = []
    temp = []
    for i in D[s:]:
        temp.append(i)
        if len(temp) == t:
            out.append(temp)
            temp = []
    for i in range(len(n_D)):
        out[i].append(n_D[i])
    return out

if __name__ == '__main__':
    #Create a dataset, igual al código compartido en Classroom
    X, y = make_blobs(n_samples=1000, n_features=50, centers=3, center_box=(-1.0, 1.0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    # Generar combinaciones para SVM, Se obtuvo por codigo compartido en Classroom
    keys_svm, values_svm = zip(*param_grid_svm.items())
    combinations_svm = [dict(zip(keys_svm, v)) for v in itertools.product(*values_svm)]

    # Numero de cores (procesos paralelos) a usar
    N_CORES = 10

    
    splits = nivelacion_cargas(combinations_svm, N_CORES) #Se divide el grid de parametros entre los procesos
    lock = multiprocess.Lock() #Se crea un lock para evitar problemas de concurrencia en la impresión de resultados

    output_file = "SVM_results.txt"  #Archivo de salida
    # Limpiar el archivo de salida
    open(output_file, "w").close()
    
    start_time = time.perf_counter() #Se inicia el tiempo de ejecución

    threads = [] #rreglo para guardar los procesos paralelos
    for i in range(N_CORES):
        threads.append(multiprocess.Process(target=evaluate_svm, args=(splits[i], X_train, y_train, X_test, y_test, lock,output_file))) #Se crean los procesos paralelos con la función evaluate_svm

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    finish_time = time.perf_counter()
    
    #Se imprime el tiempo total de ejecución
    with open(output_file, "a") as f:
        f.write(f"Tiempo total: {finish_time - start_time:.2f} segundos.\n")
    
    output_file.close()
    print(f"Total Time: {finish_time - start_time:.2f} seconds")
