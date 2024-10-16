import itertools
import multiprocess
import time
import xgboost as xgb
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

# Hyperparameter grid for XGBoost
param_grid_xgb = {
    'n_estimators': [10, 50, 100,150], # Numero de arboles
    'learning_rate': [0.01, 0.1, 0.2, 0.3], # Tasa de aprendizaje
    'max_depth': [2,4, 6, 9], # Profundidad maxima de los arboles
    'subsample': [0.5, 0.7, 0.9, 1.0], # Proporciona de muestras para entrenar
}

#Calculate precision, recall and f1 score
def analyze_precision_recall_f1(y_test, y_pred):
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    return precision, recall, f1

#Para ebvaluar el modelo con diferentes hiperparametros
def evaluate_xgb(hyperparameter_set, X_train, y_train, X_test, y_test, lock, output_file):
    for params in hyperparameter_set:
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
        precision, recall, f1 = analyze_precision_recall_f1(y_test, y_pred)
        
        # Acquiring the lock before writing to the file
        lock.acquire()
        with open(output_file, "a") as f:
            f.write(f"Parametros: {params} Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}\n")
        lock.release()

#Funcion para dividir el dataset en partes iguales - Obtenida en Classroom
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
    #Crear dataset
    X, y = make_blobs(n_samples=10000, n_features=50, centers=3, center_box=(-1.0, 1.0))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    
    #Generar todas las combinaciones de hiperparametros
    keys_xgb, values_xgb = zip(*param_grid_xgb.items())
    combinations_xgb = [dict(zip(keys_xgb, v)) for v in itertools.product(*values_xgb)]

    # NUmero de cores
    N_CORES = 10

    #Separa las combinaciones en partes iguales
    splits = nivelacion_cargas(combinations_xgb, N_CORES)
    lock = multiprocess.Lock()

    output_file = "xgboost_results.txt"  #Archivo Output

    # Limpiar el archivo de salida
    open(output_file, "w").close()

    # Start timing
    start_time = time.perf_counter()

    #Crea los procesos para evaluar el modelo con diferentes hiperparametros
    threads = [] # Lista de procesos a ejecutar en paralelo
    for i in range(N_CORES):
        threads.append(multiprocess.Process(target=evaluate_xgb, args=(splits[i], X_train, y_train, X_test, y_test, lock, output_file))) # Crea un proceso con la funcion evaluate_xgb y los argumentos correspondientes

    # Para cada proceso en la lista de procesos, se inicia
    for thread in threads:
        thread.start()

    # Para cada proceso en la lista de procesos, se espera a que termine
    for thread in threads:
        thread.join()

    # Finish timing
    finish_time = time.perf_counter()
    print(f"Total Time: {finish_time - start_time:.2f} seconds")
