import itertools
import multiprocess
import time
import xgboost as xgb
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

#Parametros para XGBoost
param_grid_xgb = {
    'n_estimators': [10, 50, 100, 150], #Numero de arboles a construir
    'learning_rate': [0.01, 0.1, 0.15, 0.2],#Tasa de aprendizaje para cada arbol
    'max_depth': [2, 4, 7, 9],#Profundidad maxima de cada arbol
    'subsample': [0.2, 0.5, 0.8, 1.0]#Fraccion de muestras a usar para entrenar cada arbol
}

#FUncion para calcular la precision, recall y f1
def analyze_precision_recall_f1(y_test, y_pred):  
    recall = recall_score(y_test, y_pred, average='weighted') #Calcula el recall 
    f1 = f1_score(y_test, y_pred, average='weighted') #Calcula el f1
    return recall, f1

#funcion para evaluar el modelo XGBoost
def evaluate_xgb(hyperparameter_set, X_train, y_train, X_test, y_test, lock, output_file):
    for params in hyperparameter_set: #iTERAR SOBRE LOS PARAMETROS
        start_time = time.perf_counter()  #tiempo de inicio
        model = xgb.XGBClassifier(**params) #Inicializar el modelo
        model.fit(X_train, y_train) #Entrenar el modelo
        y_pred = model.predict(X_test)#Predecir con el modelo
        end_time = time.perf_counter()#Tiempo de fin 

        accuracy = accuracy_score(y_test, y_pred)
        recall, f1 = analyze_precision_recall_f1(y_test, y_pred)#Calcular precision, recall y f1
        time_taken = end_time - start_time#Tiempo total

        #Escribe los resultados en el archivo de salida
        lock.acquire()
        with open(output_file, "a") as f:
            f.write(f"Parametros: {params}, Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, Tiempo: {time_taken:.4f} segundos\n")
        lock.release()

#Funcion para nivelar las cargas de trabajo - Obtenido en Classroom
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
    #Crear un conjunto de datos de prueba
    X, y = make_blobs(n_samples=1000, n_features=50, centers=3, center_box=(-1.0, 1.0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    #Generar todas las combinaciones de parametros - Obtenido del codigo compartido en Classroom
    keys_xgb, values_xgb = zip(*param_grid_xgb.items())
    combinations_xgb = [dict(zip(keys_xgb, v)) for v in itertools.product(*values_xgb)]

    #Num de cores
    N_CORES = 10

    # Nivelar las cargas de trabajo
    splits = nivelacion_cargas(combinations_xgb, N_CORES)
    lock = multiprocess.Lock()

    output_file = "xgboost_results.txt"  #Archivo de salida
    # Limpiar el archivo de salida
    open(output_file, "w").close()

    # Iniciar el temporizador
    start_time = time.perf_counter()

    # Iniciar los procesos para evaluar el modelo XGBoost
    threads = []
    for i in range(N_CORES):
        threads.append(multiprocess.Process(target=evaluate_xgb, args=(splits[i], X_train, y_train, X_test, y_test, lock, output_file))) #Crear el proceso con los parametros

    # Iniciar los procesos
    for thread in threads:
        thread.start()

    # Esperar a que todos los procesos terminen
    for thread in threads:
        thread.join()

    # Detener el temporizador
    finish_time = time.perf_counter()
    
    #EScribir el tiempo total de ejecución en el archivo de salida
    with open(output_file, "a") as f:
        f.write(f"Tiempo total: {finish_time - start_time:.2f} segundos.\n")
    
    output_file.close()
    
    print(f"Tiempo total: {finish_time - start_time:.2f} segundos.")
