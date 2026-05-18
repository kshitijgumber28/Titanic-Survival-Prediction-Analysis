# Titanic-Survival-Prediction-Analysis

## Project Overview
This project is a machine learning classification system built on the Titanic dataset to predict whether a passenger survived or not based on passenger details such as age, sex, passenger class, fare, and embarkation point.

The project includes:
- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model building
- Model evaluation
- Flask deployment

---

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Joblib

---

## Machine Learning Workflow

### Data Preprocessing
- Missing value handling using `SimpleImputer`
- Categorical encoding using `OneHotEncoder`
- Feature scaling using `StandardScaler`
- Skewness handling using Yeo-Johnson Transformation

### Feature Engineering
Features created:
- Family Size
- Is Alone
- Social Rank / Passenger Title

### Model Used
- Logistic Regression

---

## Model Evaluation

### Classification Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Cross Validation

### Confusion Matrix
Used to analyze:
- True Positives
- True Negatives
- False Positives
- False Negatives

---

## Flask Deployment
A Flask web application was created where users can:
- Enter passenger details
- Predict survival outcome in real-time

---

## Project Structure

```bash
Titanic-Survival-Prediction/
│
├── app.py
├── titanic_pipeline.pkl
├── Titanic.ipynb
├── requirements.txt
├── README.md
└── dataset/
```

---

## How to Run the Project

### 1. Clone Repository

```bash
git clone <your-repo-link>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Flask App

```bash
python app.py
```

---

## Future Improvements
- Try advanced models like Random Forest and XGBoost
- Improve frontend UI
- Deploy on Render/Heroku
- Hyperparameter tuning

---

## Author
Kshitij Gumber
