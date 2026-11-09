# Fake Job Posting Classifier

A Machine Learning-based web application that detects potentially fraudulent job postings by analyzing job descriptions using Natural Language Processing (NLP).

## Project Overview

Online job platforms contain fraudulent job advertisements that can mislead job seekers, collect personal information, or demand registration fees. Identifying these fake postings manually can be difficult and time-consuming.

This project uses Machine Learning to classify job postings as genuine or fraudulent based on their textual content.

## Problem Statement

Job seekers are increasingly exposed to fake job advertisements that promise unrealistic salaries, request upfront payments, or provide misleading employment information.

The goal of this project is to develop an automated system that helps identify potentially fraudulent job postings and makes online job searching safer.

## Solution

A Logistic Regression classification model is trained using TF-IDF text vectorization to identify patterns associated with fraudulent job advertisements.

The trained model is integrated into a Flask web application where users can paste a job description and receive a prediction along with the model's confidence score.

## Features

- Detects potentially fraudulent and genuine-looking job postings.
- Uses Natural Language Processing for text analysis.
- Displays prediction confidence percentages.
- Visual confidence progress bar.
- Input validation to prevent very short descriptions.
- Character counter for the text input.
- Clear button to reset the input.
- User-friendly Flask web interface.

## Technologies Used

- Python
- Flask
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Logistic Regression
- HTML
- CSS
- SHAP
- Joblib

## Machine Learning Workflow

1. Load and explore the job posting dataset.
2. Handle missing values.
3. Combine relevant text columns.
4. Convert text into numerical features using TF-IDF.
5. Split the dataset into training and testing sets.
6. Train and evaluate classification models.
7. Tune the Logistic Regression model.
8. Save the trained model and TF-IDF vectorizer.
9. Integrate the model into a Flask application.

## Model Evaluation

The project evaluates the classification model using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- ROC-AUC
- PR-AUC

The Logistic Regression model achieved approximately 97% accuracy, with an F1-score of 0.73 for the fraudulent class in the recorded evaluation.

## Project Structure

```text
Fake-Job-Posting-Classifier/
│
├── dataset/
│   └── fake_job_postings.csv
│
├── model/
│   ├── logistic_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── preprocess.py
├── app.py
├── README.md
└── requirements.txt
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/agnimithratheyyeth-cpu/Fake-Job-Posting-Classifier.git
```

### 2. Navigate to the project directory

```bash
cd Fake-Job-Posting-Classifier
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the Flask application

```bash
python app.py
```

### 7. Open the application

Visit:

http://127.0.0.1:5000

## How to Use

1. Open the Fake Job Posting Detector.
2. Paste a job description into the text area.
3. Click **Analyze Job Posting**.
4. View the predicted classification and confidence score.

## Limitations

- Predictions depend on the quality and characteristics of the training dataset.
- The model may misclassify some genuine or fraudulent postings.
- Confidence scores represent model-estimated probabilities and do not guarantee prediction accuracy.
- Users should independently verify employers and job offers before sharing personal information or making payments.

## Future Improvements

- Deploy the application online.
- Add prediction history.
- Improve model