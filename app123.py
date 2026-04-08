import streamlit as st
import pandas as pd
import joblib
import numpy as np

# --- STEP 1: LOAD MODELS ---
@st.cache_resource
def load_models():
    lr = joblib.load('logistic_regression_model.pkl')
    rf = joblib.load('random_forest_model.pkl')
    dt = joblib.load('decision_tree_model.pkl')
    return lr, rf, dt

try:
    lr, rf, dt = load_models()
except:
    st.error("Model files not found. Please run your export code first.")

# --- STEP 2: DEFINE MAPPINGS ---
# These match your provided logic exactly
gender_map = {'Male': 0, 'Female': 1}
smartphone_map = {'less than 2': 0, '2-4': 1, '4-6': 2, '6-8': 3, 'more than 8': 4}
approx_map = {'less than 1': 0, '1-3': 1, '3-5': 2, 'more than 5': 3}
notification_map = {'Instantly': 0, 'Every few minutes': 1, 'Every hour': 2, 'Rarely': 3}
sleep_map_time = {'less than 1': 0, '1-2': 1, '2-3': 2, 'more than 3': 3}
hour_map = {'less than 5': 0, '5-7': 1, 'more than 9': 2, '7-9': 3}
tired_map = {'no': 2, 'sometimes': 1, 'yes': 0}
study_map = {'less than 1': 0, '1-3': 1, '3-5': 2, 'more than 5': 3}
cgpa_map = {'less than 5': 0, '5-6': 1, '6-7': 2, '7-8': 3, '8-9': 4, 'more than 9': 5}
addicted_map = {'very much': 0, 'Moderately': 1, 'not at all': 2}
stressed_map = {'Never': 0, 'often': 1, 'rarely': 2}
spending_map = {'agree': 0, 'neutral': 1, 'disagree': 2}
anxious_map = {'always': 0, '2': 1, 'often': 2, 'rarely': 3, 'never': 4}
yes_no_3map = {'No': 1, 'Yes': 0} # Mapping where No=1, Yes=0
yes_no_1map = {'no': 0, 'yes': 1} # Mapping where No=0, Yes=1
yes_no_2map = {'yes': 0, 'no': 1} # Mapping where Yes=0, No=1

# --- STEP 3: USER INTERFACE ---
st.title("📊 Digital Usage Behavior Analysis")
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "Logistic Regression", "Decision Tree"))

st.subheader("Enter your Habits & Patterns")

# Organizing inputs into columns for a better layout
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("What is your age?", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    usage = st.selectbox("Daily Smartphone Usage", list(smartphone_map.keys()))
    app_cat = st.selectbox("Which category of apps do you use most?", 
                            ["Social Media", "Entertainment", "Productivity", "Gaming", "Education"])
    notif = st.selectbox("How often do you check notifications?", list(notification_map.keys()))
    sleep_before = st.selectbox("Do you use phone before sleeping?", ["Yes", "No"])
    sleep_time = st.selectbox("Hours of phone use before sleep?", list(sleep_map_time.keys()))
    wake_up = st.selectbox("Use device immediately after waking up?", ["Yes", "No"])
    daily_sleep = st.selectbox("Total sleep hours daily?", list(hour_map.keys()))

with col2:
    tired = st.selectbox("Feel tired/sleepy during the day?", list(tired_map.keys()))
    study = st.selectbox("Daily study hours?", list(study_map.keys()))
    cgpa = st.selectbox("What is your CGPA?", list(cgpa_map.keys()))
    focus = st.selectbox("Able to focus during study?", ["yes", "no"])
    stress_source = st.selectbox("Digital usage is main source of stress?", ["yes", "no"])
    addicted = st.selectbox("Feel addicted to digital devices?", list(addicted_map.keys()))
    screen_stress = st.selectbox("Stressed after long screen time?", list(stressed_map.keys()))
    mood_effect = st.selectbox("Spending time online affects mood?", list(spending_map.keys()))
    anxious = st.selectbox("Anxious without phone/internet?", list(anxious_map.keys()))

# --- STEP 4: PREDICTION LOGIC ---
if st.button("Predict Productivity Level"):
    # 1. Map simple inputs
    data = {
        'What is your age?': age,
        'Gender?': gender_map[gender],
        'How many hours per day do you use your smartphone?': smartphone_map[usage],
        'Approximate time spent per day on Social media ?(in hours)': approx_map[usage if usage in approx_map else 'less than 1'],
        'How often do you check notifications?': notification_map[notif],
        'Do you use your phone before sleeping?': yes_no_3map[sleep_before],
        'How long do you use your phone before going to sleep?(in hours)': sleep_map_time[sleep_time],
        'Do you use your device immediately after waking up?': yes_no_3map[wake_up],
        'how many hours you sleep daily?': hour_map[daily_sleep],
        'Do you feel tired or sleepy during the day?': tired_map[tired],
        'How many hours do you study daily?': study_map[study],
        'what is your cgpa?': cgpa_map[cgpa],
        'Are you able to focus while study sessions?': yes_no_1map[focus],
        'is your digital usage is main source of your stress?': yes_no_2map[stress_source],
        'Do you feel addicted to digital devices?': addicted_map[addicted],
        'Do you feel stressed after long hours of screen time?': stressed_map[screen_stress],
        'Does spending a lot of time online affect your mood?': spending_map[mood_effect],
        'Do you feel anxious when you cannot access your phone or internet?': anxious_map[anxious]
    }

    # 2. Handle One-Hot Encoding for 'Which categories of apps do you use the most?'
    # Initialize all columns to 0
    categories = ['Social Media', 'Entertainment', 'Productivity', 'Gaming', 'Education']
    for cat in categories:
        data[f'Which categories of apps do you use the most?_{cat}'] = 1 if app_cat == cat else 0

    # Convert to DataFrame
    input_df = pd.DataFrame([data])

    # 3. Model Selection
    selected_model = rf if model_choice == "Random Forest" else (lr if model_choice == "Logistic Regression" else dt)
    
    # 4. Predict
    prediction = selected_model.predict(input_df)
    
    # 5. Display Result
    st.markdown("---")
    res_map = {0: "Low Productive", 1: "Medium Productive", 2: "High Productive"}
    result_text = res_map.get(prediction[0], prediction[0])
    
    if "High" in str(result_text):
        st.success(f"Prediction: {result_text}")
    elif "Medium" in str(result_text):
        st.info(f"Prediction: {result_text}")
    else:
        st.warning(f"Prediction: {result_text}")