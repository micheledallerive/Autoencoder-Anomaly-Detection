# MNIST Autoencoder anomaly detection
This is a simple example of anomaly detection using an autoencoder. The autoencoder is trained on the MNIST dataset, using one of the digits as the anomaly.

The model decodes the input image and compares it to the original image using a loss function. If the loss is above a threshold, the image is classified as an anomaly.

## Usage
The repository contains the trained model in the folder `autoencoder`. 
The model has a 95% accuracy in recognizing the anomalies; the threshold is dynamically computed by an algorithm that finds the most effective one.

To run the model analysis, run the following command:
```
python3 analysis.py
```
The analysis will load the model and run it on the test set. 
It will then plot the threshold and the loss of each image; the images with a loss above the threshold are classified as anomalies.

The analysis script also prints the percentage of detected anomalies, undetected anomalies, incorrectly detected nominals and correctly detected nominals.

## Requirements
The requirements are listed in the `requirements.txt` file.

## References
The MNIST dataset is available at http://yann.lecun.com/exdb/mnist/.
