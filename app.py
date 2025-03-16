from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

app = Flask(__name__)

def calculate_tdee(weight, height, age, gender, activity_level):
    """Calculate Total Daily Energy Expenditure (TDEE) based on user input."""
    if gender == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    activity_multipliers = {
        "sedentary": 1.2, "light": 1.375, "moderate": 1.55, "active": 1.725, "very active": 1.9
    }
    return bmr * activity_multipliers.get(activity_level, 1.2)

# Load dataset
df = pd.read_csv("train.csv")

# Auto-detect cholesterol column
cholesterol_col = next((col for col in df.columns if "cholesterol" in col.lower()), None)

def preprocess_data(df, daily_calories=2000, meals_per_day=3, hba1c=5.0, cholesterol=150):
    """Preprocess dataset to filter meals based on user health conditions."""
    nutrient_features = ["Energy_kcal", "Protein_g", "Fat_g", "Carb_g", "Fiber_g", "Sugar_g"]
    if cholesterol_col:
        nutrient_features.append(cholesterol_col)
    
    df_filtered = df.dropna(subset=nutrient_features)
    df_filtered = df_filtered[~((df_filtered[nutrient_features] == 0).all(axis=1))]

    protein_threshold = (daily_calories * 0.10 / 4) / meals_per_day
    carb_threshold = (daily_calories * 0.45 / 4) / meals_per_day
    fat_threshold = (daily_calories * 0.35 / 9) / meals_per_day

    df_filtered["is_good_meal"] = (
        (df_filtered["Protein_g"] >= protein_threshold) &
        (df_filtered["Carb_g"] >= carb_threshold) &
        (df_filtered["Fat_g"] <= fat_threshold)
    ).astype(int)

    return df_filtered, nutrient_features

def get_ml_meal_plan(user_calories, df_filtered, model, nutrient_features,user_hba1c,user_cholesterol, num_meals=4, num_sets=1):
    """Generates meal plans suitable for all users, including those with high sugar and cholesterol."""
    df_filtered = df_filtered.drop(columns=["ID"], errors='ignore')
    df_filtered = df_filtered.dropna()
    df_filtered = df_filtered[df_filtered["Energy_kcal"] > 10]
    
    df_filtered["score"] = model.predict_proba(df_filtered[nutrient_features])[:, 1]
    df_filtered = df_filtered.sort_values(by="score", ascending=False)

    meal_sets = []
    meal_calories = user_calories / num_meals

    for _ in range(num_sets):
        meal_plan = {"Breakfast": [], "Lunch": [], "Dinner": [], "Snack": []}

        for meal in meal_plan.keys():
            selected_items = []
            remaining_calories = meal_calories

            for _, row in df_filtered.sample(frac=1).iterrows():
                if row["Energy_kcal"] <= remaining_calories:
                    if user_hba1c >= 5.7 and row["Sugar_g"] > 5:
                        continue  # Skip high sugar foods only for high HbA1c users
                    if user_cholesterol >= 200 and cholesterol_col and row[cholesterol_col] > 75:
                        continue  # Skip high cholesterol foods only for high cholesterol users
                    selected_items.append(row[["Descrip", "Energy_kcal", "Protein_g", "Fat_g", "Carb_g", "Fiber_g", "Sugar_g"]])
                    remaining_calories -= row["Energy_kcal"]
                if remaining_calories <= 0:
                    break

            meal_plan[meal] = pd.DataFrame(selected_items)

        meal_sets.append(meal_plan)

    return meal_sets

@app.route('/meal_plan', methods=['POST'])
def meal_plan():
    data = request.get_json()

   
    user_weight = float(data['weight'])
    user_height = float(data['height'])
    user_age = int(data['age'])
    user_gender = data['gender'].strip().lower()
    user_activity = data['activity_level'].strip().lower()
    user_hba1c = float(data['hba1c'])
    user_cholesterol = float(data['cholesterol'])

    hba1c_status = "Normal" if user_hba1c < 5.7 else "Prediabetes/Diabetes"
    cholesterol_status = "Normal" if user_cholesterol < 200 else "High Cholesterol"

    tdee_calories = calculate_tdee(user_weight, user_height, user_age, user_gender, user_activity)

    df_filtered, nutrient_features = preprocess_data(df, daily_calories=tdee_calories, hba1c=user_hba1c, cholesterol=user_cholesterol)

    X = df_filtered[nutrient_features]
    y = df_filtered["is_good_meal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize XGBoost model with base_score=0.5
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        base_score=0.5
    )
    model.fit(X_train, y_train)

    meal_plans = get_ml_meal_plan(tdee_calories, df_filtered, model, nutrient_features,user_hba1c,user_cholesterol)
    
    response = {
        "tdee_calories": tdee_calories,
        "hba1c_status": hba1c_status,
        "cholesterol_status": cholesterol_status,
        "meal_plans": []
    }

    for i, meal_plan in enumerate(meal_plans, start=1):
        meal_plan_dict = {}
        for meal, items in meal_plan.items():
            meal_plan_dict[meal] = items.to_dict(orient='records')
        response["meal_plans"].append(meal_plan_dict)

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)