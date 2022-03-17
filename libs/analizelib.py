import pandas as pd
import numpy as np

def find_threshold(reconstructions, x_set, loss_function, percentile=94):
    """Calculates the threshold to recognize anomalies using a percentile"""
    errors = loss_function(reconstructions, x_set)
    return np.percentile(errors, percentile)

def get_predictions(predictions, x_test, threshold, loss_function):
    """Calculates the prediction of the model for each image of the set (0.0 if it's anomal, 1.0 if it's normal)"""
    errors = loss_function(predictions, x_test)
    mask = pd.Series(errors) >= threshold
    return mask.map(lambda x: 0.0 if x == True else 1.0)
  
def analize_predictions(predictions, labels):
    """Analizes the predictions to calculate the accuracy of the model"""
    detected_anomalies, undetected_anomalies, incorrecly_detected_anomalies, correctly_detected_nominal=0,0,0,0
    num_anomalies, num_nominals = 0,0
    for i, prediction in enumerate(predictions):
        if labels[i] == "Anomal":
            num_anomalies+=1
            if prediction==0.0:
                detected_anomalies+=1
            else:
                undetected_anomalies+=1
        else:
            num_nominals+=1
            if prediction==0.0:
                incorrecly_detected_anomalies+=1
            else:
                correctly_detected_nominal+=1
    return (detected_anomalies, undetected_anomalies, incorrecly_detected_anomalies, correctly_detected_nominal, num_anomalies, num_nominals)