🩺 Diabetic Retinopathy Detection using Random Forest

This project implements a Machine Learning approach to detect Diabetic Retinopathy (DR) from retinal fundus images. The system applies image preprocessing and a Random Forest classifier to predict whether a given retinal image shows signs of DR (DR or No_DR).

🚀 Features

Image Preprocessing: Resizing, normalization, and feature extraction from retinal fundus images.

Random Forest Classifier: Ensemble-based ML model for binary classification.

Evaluation Metrics: Accuracy, precision, recall, F1-score, and confusion matrix.

Prediction: Supports single image prediction via trained model.

Extendable: Can be integrated with a web app (Flask/Django) for interactive diagnosis.

🛠️ Tech Stack

Programming Language: Python

Libraries: scikit-learn, NumPy, Pandas, OpenCV, Matplotlib, Seaborn

Environment: Jupyter Notebook

📊 Workflow

Dataset Preparation

Collected retinal fundus images from public datasets.

Preprocessed images (resizing, flattening, normalization).

Feature Engineering

Extracted pixel intensity features.

Created structured dataset for Random Forest training.

Model Training

Trained Random Forest classifier on preprocessed dataset.

Tuned hyperparameters for better accuracy.

Model Evaluation

Evaluated with classification report and confusion matrix.

Prediction

Load trained model (.pkl file).

Run predictions on new retinal images.

📈 Results

Achieved high accuracy with balanced precision and recall.

Random Forest proved effective for binary classification of DR vs No_DR.

🔮 Future Enhancements

Upgrade to multi-class classification (different DR severity levels).

Add Grad-CAM visualizations (if deep learning is used in future).

Deploy as a Flask/Django web app for real-time usage.
