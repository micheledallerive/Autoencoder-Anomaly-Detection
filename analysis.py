import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from libs.analizelib import analize_predictions,get_predictions
from libs.setutils import load_sets

LOSS_FUNCTION = tf.keras.losses.mean_squared_error

def plot_errors(test_losses, testing_labels, threshold):
    """Plots the reconstruction errors of each image"""
    data_frame = pd.DataFrame({'error': test_losses, "type": testing_labels})
    groups = data_frame.groupby('type')
    fig, axis = plt.subplots()
    for name, group in groups:
        axis.plot(group.index, group["error"], marker='o', ms=2.5, zorder=5 if name=="Anomal" else 1, 
                linestyle='', label=name, color="red" if name=="Anomal" else "green")
    axis.hlines(threshold,axis.get_xlim()[0],axis.get_xlim()[1],colors="black",zorder=50,label='Threshold')
    axis.legend()
    plt.title("Reconstruction error for type of data")
    plt.ylabel("Reconstruction error")
    plt.xlabel("Index of the image")
    plt.show()

def load_threshold(path):
    """Loads the model threshold from its file"""
    with open(os.path.join(path,"threshold.txt")) as thresh_file:
        return float(thresh_file.readline())
    
def main():
    print("Loading the model...")
    model = keras.models.load_model("./autoencoder")
    anomaly = [8] # the model was trained to recognize 8s as anomalies
    ((x_train, x_test),\
        (anomaly_set_train, nominal_set_train, train_labels),\
        (anomaly_set_test, nominal_set_test, test_labels)) = load_sets(anomaly)

    test_predictions = model.predict(x_test)
    test_losses = LOSS_FUNCTION(x_test, test_predictions).numpy()
    threshold = load_threshold("./autoencoder")

    preds = get_predictions(test_predictions, x_test, threshold, LOSS_FUNCTION)

    detected_anomalies,\
    undetected_anomalies,\
    incorrecly_detected_anomalies,\
    correctly_detected_nominal,\
    num_anomalies,\
    num_nominals = analize_predictions(preds, test_labels)

    print(f'Detected anomalies: {detected_anomalies}/{num_anomalies} ({round(detected_anomalies/num_anomalies*100,2)}%)')
    print(f'Undetected anomalies: {undetected_anomalies}/{num_anomalies} ({round(undetected_anomalies/num_anomalies*100,2)}%)')
    print(f'Incorrectly detected anomalies: {incorrecly_detected_anomalies}/{num_nominals} ({round(incorrecly_detected_anomalies/num_nominals*100,2)}%)')
    print(f'Correctly detected nominals: {correctly_detected_nominal}/{num_nominals} ({round(correctly_detected_nominal/num_nominals*100,2)}%)')

    plot_errors(test_losses, test_labels, threshold)


if __name__ == "__main__":
    main()