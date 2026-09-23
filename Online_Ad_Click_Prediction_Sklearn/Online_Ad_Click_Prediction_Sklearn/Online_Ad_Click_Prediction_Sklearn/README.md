# Online Ad Click Prediction using Python + Scikit-learn

## Project Overview
This machine learning project predicts whether a user is likely to click an online advertisement based on user behavior, device, ad position, income, and previous clicks.

## Technology Used
- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Joblib
- Matplotlib

## Files
- `data/online_ad_clicks.csv` - Synthetic dataset
- `train_model.py` - Trains and evaluates the model
- `predict.py` - Predicts whether a user will click an ad
- `online_ad_click_model.pkl` - Trained model
- `feature_importance.png` - Feature importance chart
- `requirements.txt` - Required libraries

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python train_model.py
```

### 3. Predict a new user
```bash
python predict.py
```

## Target
- `1` = Clicked
- `0` = Not Clicked

## Note
The dataset is synthetic and intended for educational purposes only. Real advertising systems should follow privacy, consent, fairness, and applicable data-protection requirements.
