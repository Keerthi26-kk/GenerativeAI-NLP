AI-Powered Resume Screener

Overview

This project is an AI-powered Resume Screener that uses Natural Language Processing (NLP) and Machine Learning (ML) to classify resumes into different job categories. It utilizes pipelines and GridSearchCV for efficient preprocessing and model optimization.

Features

Automated resume classification based on text content.

TF-IDF Vectorization for text feature extraction.

Pipeline Integration for seamless data transformation and modeling.

GridSearchCV Optimization to find the best hyperparameters.

Random Forest Classifier for accurate classification.

Installation

1. Clone the Repository

git clone https://github.com/yourusername/resume-screener.git
cd resume-screener

2. Install Dependencies

pip install -r requirements.txt

Usage

1. Train the Model

python train.py

This will:

Clean and preprocess the dataset.

Train the model using a pipeline with TF-IDF and Random Forest.

Optimize hyperparameters using GridSearchCV.

Save the best model.

2. Classify a Resume

Run the script to predict a job category based on resume text:

python classify.py "Experienced data scientist with expertise in Python, machine learning, and big data."

Output example:

Predicted Category: Data Scientist

Dataset

The dataset consists of resume text samples labeled with job categories, such as:

Data Scientist

Software Engineer

Marketing Specialist

HR Manager

Cybersecurity Analyst

Model Pipeline

The machine learning pipeline consists of:

TF-IDF Vectorizer: Converts resume text into numerical features.

Random Forest Classifier: Classifies resumes into job categories.

GridSearchCV: Optimizes hyperparameters for better accuracy.

Hyperparameter Tuning

GridSearchCV optimizes the following parameters:

param_grid = {
    'vectorizer__max_features': [3000, 5000],
    'vectorizer__ngram_range': [(1,1), (1,2)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

Results

The trained model achieves high accuracy in resume classification. Evaluation metrics include:

Accuracy Score

Precision, Recall, and F1-Score

Future Enhancements

Add deep learning models (BERT, LSTM) for better NLP performance.

Expand dataset with more resume samples.

Deploy as a web API for real-time classification.

License

This project is open-source under the MIT License.
