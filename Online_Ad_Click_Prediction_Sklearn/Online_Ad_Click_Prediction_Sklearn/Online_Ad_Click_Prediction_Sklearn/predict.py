import joblib
import pandas as pd

model = joblib.load("online_ad_click_model.pkl")

print("Enter online advertisement details")

age = int(input("Age: "))
daily_time = float(input("Daily time on site in minutes: "))
pages = int(input("Pages visited: "))
income = float(input("Annual income: "))
internet_usage = float(input("Internet usage hours per day: "))
gender = input("Gender (Male/Female): ")
device = input("Device (Mobile/Desktop/Tablet): ")
ad_position = input("Ad position (Top/Middle/Bottom): ")
previous_clicks = int(input("Previous ad clicks: "))

data = pd.DataFrame([{
    "age": age,
    "daily_time_on_site": daily_time,
    "pages_visited": pages,
    "annual_income": income,
    "internet_usage_hours": internet_usage,
    "gender": gender,
    "device": device,
    "ad_position": ad_position,
    "previous_clicks": previous_clicks
}])

prediction = model.predict(data)[0]
confidence = model.predict_proba(data).max()

if prediction == 1:
    print("\nPrediction: User likely to CLICK the advertisement")
else:
    print("\nPrediction: User likely to NOT CLICK the advertisement")

print(f"Confidence: {confidence:.2%}")
