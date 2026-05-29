#  CIFAR-10 MLOps Project (MLflow + W&B + Streamlit + Grad-CAM)

##  Overview

This project demonstrates a complete **MLOps pipeline** for image classification using the CIFAR-10 dataset. It integrates:

-  PyTorch CNN model
-  Experiment tracking (MLflow + Weights & Biases)
-  Interactive Streamlit dashboard
-  Model interpretability (Grad-CAM)
-  Multiple experiment runs for comparison

The goal is to build a **reproducible and trackable ML system** with full experiment lifecycle support.

##   Requirements

streamlit

mlflow

torch

torchvision

numpy

pandas

matplotlib

seaborn

scikit-learn

pyyaml

grad-cam

wandb

opencv-python

##   Installation

python3 -m venv venv

source venv/bin/activate

pip install -r requirements.txt

##   Start MLflow UI

mlflow ui --backend-store-uri file:./mlruns --port 5001

##    Train the model (multiple runs)

python3 src/train.py

##    Start Streamlit dashboard

streamlit run app.py
