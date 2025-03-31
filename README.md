 AI-Powered Resume Screener

 Overview

Finding the right candidate for a job is challenging. This AI-powered Resume Screener simplifies the process by leveraging Natural Language Processing (NLP) and Machine Learning (ML) to automatically classify resumes into different job categories.

✅ Seamless AI-driven Resume Screening
✅ Automated Text Processing with Pipelines
✅ Hyperparameter Optimization with GridSearchCV
✅ Fast, Efficient & Scalable!

 Features

📌 Automated Resume Classification based on text content.

📌 TF-IDF Vectorization for text feature extraction.

📌 Pipeline Integration for seamless data transformation and modeling.

📌 GridSearchCV Optimization to find the best hyperparameters.

📌 Random Forest Classifier for accurate classification.

 Installation

1️⃣ Clone the Repository

git clone https://github.com/yourusername/resume-screener.git
cd resume-screener

2️⃣ Install Dependencies

pip install -r requirements.txt

🚀 Usage

1️⃣ Train the Model

Run the following command to train the model:

python train.py

This will:

✅ Clean and preprocess the dataset.

✅ Train the model using a TF-IDF + Random Forest pipeline.

✅ Optimize hyperparameters using GridSearchCV.

✅ Save the best model for later use.

2️⃣ Classify a Resume

Use the trained model to predict job categories:

python classify.py "Experienced data scientist with expertise in Python, machine learning, and big data."

Example Output:

Predicted Category: Data Scientist

📂 Dataset

This project uses a labeled dataset of resumes, categorized into different job roles such as:

Software Engineer

 Data Scientist

 Marketing Specialist

 HR Manager

 Cybersecurity Analyst

🔄 Model Pipeline

Our model workflow consists of the following steps:
1️⃣ TF-IDF Vectorizer: Converts resume text into numerical features.
2️⃣ Random Forest Classifier: Predicts job categories based on extracted features.
3️⃣ GridSearchCV: Tunes hyperparameters to improve accuracy.

 Hyperparameter Tuning

To improve model accuracy, we use GridSearchCV with the following parameter grid:

param_grid = {
    'vectorizer__max_features': [3000, 5000],
    'vectorizer__ngram_range': [(1,1), (1,2)],
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [10, 20, None],
    'classifier__min_samples_split': [2, 5],
    'classifier__min_samples_leaf': [1, 2]
}

 Results

 High Accuracy Resume Classification with performance evaluation using:

● Accuracy Score
● Precision, Recall, and F1-Score

 Future Enhancements

🔹 Deep Learning Integration: Add BERT, LSTMs for better NLP performance.🔹 Larger Dataset: Improve generalization by expanding training data.🔹 Web API Deployment: Convert this into an interactive tool for businesses.

📜 License

This project is open-source under the MIT License. Use it freely and contribute!

