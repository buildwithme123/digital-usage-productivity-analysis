import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# --- STEP 1: LOAD ASSETS ---
@st.cache_resource
def load_assets():
    try:
        lr = joblib.load('logistic_regression.pkl')
        rf = joblib.load('random_forest.pkl')
        dt = joblib.load('decision_tree.pkl')
        # Load background data for SHAP (Ensure this file is in your GitHub)
        bg_data = pd.read_csv('train_reference.csv') 
        return lr, rf, dt, bg_data
    except Exception as e:
        st.error(f"Asset loading failed: {e}. Ensure .pkl and .csv files are in the repository.")
        return None, None, None, None

lr, rf, dt, X_train_bg = load_assets()

# --- STEP 2: MAPPINGS & DICTIONARIES ---
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
anxious_map = {'always': 0, 'often': 2, 'rarely': 3, 'never': 4}
yes_no_3map = {'No': 1, 'Yes': 0}
yes_no_1map = {'no': 0, 'yes': 1}
yes_no_2map = {'yes': 0, 'no': 1}

# --- STEP 3: SHAP FUNCTIONS ---
def manual_shap_single(model, sample, background, n_repeats=50):
    sample_arr = sample.values.flatten().astype(float)
    n_feat = len(sample_arr)
    shapley = np.zeros(n_feat)
    pred_proba = model.predict_proba(sample)[0]
    target_class = int(np.argmax(pred_proba))
    bg = background.sample(n=min(50, len(background)), random_state=42)

    for _ in range(n_repeats):
        perm = np.random.permutation(n_feat)
        prev_val = model.predict_proba(bg).mean(axis=0)
        for k in range(n_feat):
            idx = perm[k]
            s_with = bg.copy()
            for j in perm[:k+1]:
                s_with.iloc[:, j] = sample_arr[j]
            pred_with = model.predict_proba(s_with).mean(axis=0)
            shapley[idx] += (pred_with - prev_val)[target_class]
            prev_val = pred_with
    return shapley / n_repeats, target_class, pred_proba

# --- STEP 4: USER INTERFACE ---
st.title("📊 Digital Usage Behavior Analysis")
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ("Random Forest", "Logistic Regression", "Decision Tree"))

st.subheader("Enter your Habits & Patterns")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("What is your age?", 10, 100, 20)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    usage = st.selectbox("Daily Smartphone Usage", list(smartphone_map.keys()))
    notif = st.selectbox("How often do you check notifications?", list(notification_map.keys()))
    sleep_before = st.selectbox("Do you use phone before sleeping?", ["Yes", "No"])
    sleep_time = st.selectbox("Hours before going to sleep?", list(sleep_map_time.keys()))
    wake_up = st.selectbox("Immediately after waking up?", ["Yes", "No"])
    daily_sleep = st.selectbox("Total sleep hours daily?", list(hour_map.keys()))

with col2:
    tired = st.selectbox("Feel tired during the day?", list(tired_map.keys()))
    study_hrs = st.selectbox("Daily study hours?", list(study_map.keys()))
    cgpa = st.selectbox("What is your CGPA?", list(cgpa_map.keys()))
    focus = st.selectbox("Able to focus while studying?", ["yes", "no"])
    stress_source = st.selectbox("Is usage your main source of stress?", ["yes", "no"])
    addicted = st.selectbox("Feel addicted to devices?", list(addicted_map.keys()))
    screen_stress = st.selectbox("Stressed after long screen time?", list(stressed_map.keys()))
    mood_effect = st.selectbox("Online time affects mood?", list(spending_map.keys()))
    anxious = st.selectbox("Anxious without internet?", list(anxious_map.keys()))

# --- STEP 5: PREDICTION & ANALYSIS ---
if st.button("Generate My Analysis"):
    # 1. Map inputs to full feature set
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
        'How many hours do you study daily?': study_map[study_hrs],
        'what is your cgpa?': cgpa_map[cgpa],
        'Are you able to focus while study sessions?': yes_no_1map[focus],
        'is your digital usage is main source of your stress?': yes_no_2map[stress_source],
        'Do you feel addicted to digital devices?': addicted_map[addicted],
        'Do you feel stressed after long hours of screen time?': stressed_map[screen_stress],
        'Does spending a lot of time online affect your mood?': spending_map[mood_effect],
        'Do you feel anxious when you cannot access your phone or internet?': anxious_map[anxious]
    }
    
    input_df = pd.DataFrame([data])
    selected_model = rf if model_choice == "Random Forest" else (lr if model_choice == "Logistic Regression" else dt)
    
    with st.spinner('Analyzing patterns...'):
        shap_vals, tgt_cls, probas = manual_shap_single(selected_model, input_df, X_train_bg)
        
    # Results Display
    st.markdown("---")
    res_map = {0: "Low Productive", 1: "Medium Productive", 2: "High Productive"}
    st.header(f"Result: {res_map[tgt_cls]}")
    
    c1, c2 = st.columns(2)
    c1.metric("Confidence", f"{probas[tgt_cls]*100:.1f}%")
    
    # SHAP Factors
    st.subheader("Key Factors Driving Your Result")
    local_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP': shap_vals}).sort_values('SHAP', key=abs, ascending=False)

    for i, row in local_df.head(5).iterrows():
        label = row['Feature']
        impact = "▲ Positive" if row['SHAP'] > 0 else "▼ Negative"
        color = "green" if row['SHAP'] > 0 else "red"
        st.write(f"**{label}** (:{color}[{impact} Impact])")
        st.progress(min(abs(row['SHAP'] * 10), 1.0))

    # Final Recommendation
    if tgt_cls == 2:
        st.success("🌟 Highly Productive! Your habits are excellent.")
    elif tgt_cls == 1:
        st.info("⚠️ Medium Productivity. Try reducing screen time to improve.")
    else:
        st.warning("🚨 Low Productivity. We suggest setting strict phone limits during study.")
