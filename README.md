# online-Ad-click-prediction-Sklearn
Online Ad Click Prediction using Python and Scikit-learn — a machine learning project that predicts whether a user is likely to click an online advertisement based on user behavior, demographics, device type, ad placement, and historical click activity.
# Online Ad Click Prediction using Python and Scikit-learn

## Project Overview

Online Ad Click Prediction is a machine learning project that predicts whether a user is likely to click an online advertisement based on user behavior, demographics, device information, ad placement, and previous advertising interactions.

The project uses a Random Forest Classifier with a preprocessing pipeline to handle both numerical and categorical features. It also includes a Streamlit web application for interactive predictions, batch impression screening, and feature-importance analysis.

The dataset contains 700 records and 9 input features, with `clicked` as the target variable.

## Objectives

* Predict whether a user will click an online advertisement.
* Analyze user engagement and advertising-related behavior.
* Handle categorical and numerical features using Scikit-learn preprocessing.
* Build and evaluate a Random Forest classification model.
* Provide probability-based click predictions.
* Visualize important features influencing the prediction.
* Provide an interactive Streamlit interface for real-time predictions.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Random Forest
* Joblib
* Matplotlib
* Streamlit

## Dataset

The project uses the following features:

| Feature                | Description                                                      |
| ---------------------- | ---------------------------------------------------------------- |
| `age`                  | Age of the user                                                  |
| `daily_time_on_site`   | Daily time spent on the website in minutes                       |
| `pages_visited`        | Number of pages visited during a session                         |
| `annual_income`        | User's annual income                                             |
| `internet_usage_hours` | Daily internet usage in hours                                    |
| `gender`               | User gender                                                      |
| `device`               | Device used by the user                                          |
| `ad_position`          | Position of the advertisement                                    |
| `previous_clicks`      | Number of previous advertisement clicks                          |
| `clicked`              | Target variable indicating whether the advertisement was clicked |

### Target Variable

* `1` = Advertisement clicked
* `0` = Advertisement not clicked

The dataset is synthetic and is intended for educational and machine-learning demonstration purposes.

## Machine Learning Workflow

The project follows these steps:

1. Load the dataset using Pandas.
2. Separate input features and target variable.
3. Identify numerical and categorical features.
4. Apply One-Hot Encoding to categorical features.
5. Keep numerical features in their original form.
6. Split the dataset into training and testing sets.
7. Train a Random Forest Classifier.
8. Evaluate the model using accuracy and classification metrics.
9. Save the trained model using Joblib.
10. Generate a feature-importance visualization.
11. Use the trained model for new predictions.

## Model

The project uses a **Random Forest Classifier** with:

* 200 decision trees
* `random_state=42`
* Balanced class weights

A Scikit-learn `Pipeline` is used to combine preprocessing and model training into a single workflow.

## Project Structure

```text
Online_Ad_Click_Prediction_Sklearn/
│
├── data/
│   └── online_ad_clicks.csv
│
├── app.py
├── predict.py
├── train_model.py
├── online_ad_click_model.pkl
├── feature_importance.png
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <your-repository-url>
cd Online_Ad_Click_Prediction_Sklearn
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Train the Model

Run the training script:

```bash
python train_model.py
```

This will:

* Train the Random Forest model.
* Display model accuracy.
* Display the classification report.
* Save the trained model as `online_ad_click_model.pkl`.
* Generate `feature_importance.png`.

## Make a Prediction

Run:

```bash
python predict.py
```

The program asks for information such as:

* Age
* Daily time on site
* Pages visited
* Annual income
* Internet usage
* Gender
* Device type
* Ad position
* Previous ad clicks

It then predicts whether the user is likely to click the advertisement and displays the prediction confidence.

## Streamlit Application

The project also provides an interactive Streamlit dashboard.

Run:

```bash
streamlit run app.py
```

The application provides three main sections:

### 1. Impression CTR Predictor

Users can enter engagement, demographic, device, and advertising-placement information to generate a predicted click likelihood.

### 2. Batch Impressions Screening

A CSV file containing advertisement impressions can be uploaded for batch prediction and analysis.

### 3. Feature Importance & Diagnostics

The application provides feature-importance information to help understand which input variables contribute to the model's predictions.

## Feature Importance

The project generates a `feature_importance.png` visualization using the feature importance values obtained from the Random Forest model.

This helps identify which user behavior and advertising characteristics have the greatest influence on the model's predictions.

## Model Evaluation

The training script evaluates the model using:

* Accuracy
* Classification Report
* Precision
* Recall
* F1-score

The dataset is split using an 80/20 train-test split with stratification to maintain the class distribution.

## Applications

This project demonstrates how machine learning can be applied to:

* Online advertising analysis
* Click-through prediction
* User engagement analysis
* Advertisement targeting research
* Digital marketing analytics
* Programmatic advertising simulations

## Limitations

* The dataset is synthetic and may not represent real-world advertising behavior.
* Model performance depends on the quality and representativeness of the training data.
* Predictions should not be treated as guaranteed user behavior.
* Real advertising systems require appropriate privacy, consent, fairness, and data-protection practices.

## Future Improvements

Possible improvements include:

* Testing additional classification algorithms.
* Hyperparameter tuning.
* Cross-validation.
* ROC-AUC and precision-recall analysis.
* Model explainability using SHAP.
* Larger and more realistic datasets.
* Model performance monitoring.
* Deployment using a cloud platform.
* Adding automated batch prediction reports.


