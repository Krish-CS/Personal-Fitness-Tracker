import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import time

# Load datasets
exercise_file = "exercise.csv"
calories_file = "calories.csv"

exercise_df = pd.read_csv(exercise_file)
calories_df = pd.read_csv(calories_file)

# Merge datasets
merged_df = pd.merge(exercise_df, calories_df, on="User_ID")
merged_df.drop(columns=["User_ID"], inplace=True)

# Compute BMI and encode Gender
merged_df["BMI"] = merged_df["Weight"] / (merged_df["Height"] / 100) ** 2
merged_df.drop(columns=["Height", "Weight"], inplace=True)
merged_df["Gender_male"] = merged_df["Gender"].map({"male": 1, "female": 0})
merged_df.drop(columns=["Gender"], inplace=True)

# Define features and target
X = merged_df[["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male"]]
y = merged_df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Streamlit UI
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")

# Custom button styling (keeping the original color)
st.markdown(
    """
    <style>
        .stButton>button {
            background: linear-gradient(to right, #FF9800, #FF5722);
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 12px;
            box-shadow: 3px 3px 10px rgba(0,0,0,0.3);
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Layout
col1, col2 = st.columns([1.2, 2], gap="large")

with col1:
    st.header("User Input Parameters:")
    age = st.slider("Age:", 10, 100, 30)
    bmi = st.slider("BMI:", 15.0, 40.0, 20.0)
    duration = st.slider("Duration (min):", 0, 35, 15)
    heart_rate = st.slider("Heart Rate:", 60, 130, 80)
    body_temp = st.slider("Body Temperature (°C):", 36.0, 42.0, 38.0)
    gender = st.radio("Gender:", ("Male", "Female"))
    gender_male = 1 if gender == "Male" else 0

with col2:
    st.title("🏋️ Personal Fitness Tracker")
    st.write("Enter your parameters and get your **calories burned prediction**.")

    table_data = pd.DataFrame([[age, bmi, duration, heart_rate, body_temp, gender_male]],
                              columns=["Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Gender_male"])
    st.subheader("Your Parameters:")
    st.dataframe(table_data, use_container_width=True)

    if st.button("Predict Calories Burnt"):
        with st.spinner('🔄 Computing results...'):
            time.sleep(2)  # Simulate loading time

        input_data = np.array([[age, bmi, duration, heart_rate, body_temp, gender_male]])
        predicted_calories = model.predict(input_data)[0]

        st.toast("🔥 Prediction Ready!")

        st.subheader("Prediction:", anchor=False)
        
        # Gauge Meter Visualization
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=predicted_calories,
            title={'text': "Calories Burned"},
            gauge={
                'axis': {'range': [None, 500]},
                'bar': {'color': "#FF9800"},
                'steps': [
                    {'range': [0, 200], 'color': "#FFF3E0"},
                    {'range': [200, 400], 'color': "#FFE0B2"}
                ],
            }
        ))
        st.plotly_chart(fig)
        
        st.success(f"🔥 {predicted_calories:.2f} kilocalories")
        
        # Bar Chart Visualization
        st.subheader("Calories Burned Breakdown")
        chart_data = pd.DataFrame({
            "Category": ["Predicted Calories", "Average Calories Burned"],
            "Calories": [predicted_calories, merged_df["Calories"].mean()]
        })
        st.bar_chart(chart_data.set_index("Category"))
        
        # Pie Chart Visualization
        st.subheader("Calorie Contribution")
        fig_pie, ax_pie = plt.subplots(figsize=(4, 4))
        ax_pie.pie([predicted_calories, merged_df["Calories"].mean()], labels=["Predicted", "Average"], autopct='%1.1f%%', colors=["#FF5722", "#FFC107"])
        st.pyplot(fig_pie)
        
        # Health Tip Section
        st.subheader("Health Tip:")
        st.info("🔹 Try increasing your workout duration for better calorie burn!" if predicted_calories < 100 else "🔹 Great job! Keep up the high-intensity workouts!")

        # Estimated Fat Burn
        st.subheader("Estimated Fat Burned:")
        st.write(f"🔥 **{predicted_calories / 3500:.4f} lbs lost**")

        # Suggested Exercises
        st.subheader("Suggested Exercises:")
        st.write("🏊 Swimming, 🚴 Cycling, 🚶 Walking (Low Impact)" if bmi > 30 else "🏃 Running, 🏋️ Weight Lifting, 🏊 Swimming")

        # Similar Results Section
        st.subheader("Similar Results:")
        similar_results = merged_df.sample(5)
        st.dataframe(similar_results, use_container_width=True)

