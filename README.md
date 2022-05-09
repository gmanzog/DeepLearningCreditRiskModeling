# Deep Learning Credit Risk Modeling
Replication codes for Deep Learning Credit Risk Modeling by Manzo, Qiao

This repository contains the Python APIs to replicate the paper "Deep Learning Credit Risk Modeling," Gerardo Manzo and Xiao Qiao, The Journal of Fixed Income Fall 2021, jfi.2021.1.121; DOI: https://doi.org/10.3905/jfi.2021.1.121

![alt text](https://github.com/gmanzog/DeepLearningCreditRiskModeling/image.jpg?raw=true)

# Abstract
This article demonstrates how deep learning can be used to price and calibrate models of credit risk. Deep neural networks can learn structural and reduced-form models with high degrees of accuracy. For complex credit risk models with no closed-form solutions available, deep learning offers a conceptually simple and more efficient alternative solution. This article proposes an approach that combines deep learning with the unscented Kalman filter to calibrate credit risk models based on historical data; this strategy attains an in-sample R-squared of 98.5% for the reduced-form model and 95% for the structural model.

# How to run the codes
In each folder, the main file's name starts with 'master_' and folders are organized as follows (in order):
Markup : 
* model_simulator: simulate data from pricing models
* dnn_training: after the data is simulated, train each model
* model_calibration: calibrate models to real data using trained DNN

Finally, the folder 'data' contains all the generated data and trained models.
