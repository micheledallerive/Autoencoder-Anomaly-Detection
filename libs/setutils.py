from tensorflow import keras
from typing import Tuple, List, AnyStr
import numpy as np
def load_sets(anomaly_set):
    """Loads the MNIST set using keras library"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

    anomaly_set_train, nominal_set_train, train_labels = split_set(x_train, y_train, anomaly_set)
    anomaly_set_test, nominal_set_test, test_labels = split_set(x_test, y_test, anomaly_set)
    return ((x_train,x_test), (anomaly_set_train, nominal_set_train, train_labels), (anomaly_set_test, nominal_set_test, test_labels))

def split_set(x_set: np.array, y_set: np.array, anomalies: List[int]) -> Tuple[np.array, np.array, List[AnyStr]]:
    """Splits the give x_set into an anomaly_set and a nominal_set"""
    anomaly_set, nominal_set, labels = [],[],[]
    for i in range(len(x_set)):
        if(y_set[i] in anomalies):
            anomaly_set.append(x_set[i])
            labels.append("Anomal")
        else:
            nominal_set.append(x_set[i])
            labels.append("Normal")
    return np.array(anomaly_set), np.array(nominal_set), labels