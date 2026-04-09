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
        # Ensure you have uploaded 'train_reference.csv' to your GitHub
        bg_data = pd.read_csv('train_reference.csv') 
        return lr, rf, dt, bg_data
    except Exception as e:
        st.error(f"Asset loading failed: {e}. Check GitHub for .pkl and .csv files.")
        return None, None, None, None

lr, rf, dt, X_train_bg = load_assets()

# --- STEP 2: DATA DICTIONARIES ---
FEATURE_LABELS = {
    'What is your age?': 'Age',
    'Gender?': 'Gender',
    'How many hours per day do you use your smartphone?': 'Daily Smartphone Usage (hrs)',
    'Approximate time spent per day on Social media ?(in hours)': 'Daily Social Media Time (hrs)',
    'How often do you check notifications?': 'Notification Checking Frequency',
    'Do you use your phone before sleeping?': 'Phone Use Before Sleep',
    'How long do you use your phone before going to sleep?(in hours)': 'Pre-Sleep Phone Duration (hrs)',
    'Do you use your device immediately after waking up?': 'Phone Use on Waking Up',
    'how many hours you sleep daily?': 'Daily Sleep (hrs)',
    'Do you feel tired or sleepy during the day?': 'Daytime Tiredness/Sleepiness',
    'How many hours do you study daily?': 'Daily Study Hours',
    'what is your cgpa?': 'CGPA',
    'Are you able to focus while study sessions?': 'Focus During Study',
    'is your digital usage is main source of your stress?': 'Digital Usage = Main Stress Source',
    'Do you feel addicted to digital devices?': 'Device Addiction Level',
    'Do you feel stressed after long hours of screen time?': 'Stress After Long Screen Time',
    'Does spending a lot of time online affect your mood?': 'Online Time Affects Mood',
    'Do you feel anxious when you cannot access your phone or internet?': 'Phone-Separation Anxiety'
}

CLASS_NAMES = ['Low Productive', 'Medium Productive', 'High Productive']
CLASS_EMOJIS = ['🔴', '🟡', '🟢']

SUGGESTIONS = {
    'Daily Smartphone Usage (hrs)': {
        'high': " Your smartphone usage is high. Try setting daily screen-time limits (e.g., under 3 hrs) using built-in phone tools.",
        'low':  " Your smartphone usage is well-controlled. Keep it up!"
    },
    
    'Notification Checking Frequency': {
        'high': " Frequent notification checks break your focus. Switch to 'Do Not Disturb' during study hours.",
        'low':  " You check notifications infrequently — great for maintaining deep focus."
    },
    'Phone Use Before Sleep': {
        'high': " Using your phone before sleep disrupts melatonin production. Try stopping screens 30–60 min before bed.",
        'low':  " Avoiding pre-sleep phone use supports better sleep quality."
    },
    'Pre-Sleep Phone Duration (hrs)': {
        'high': " Long phone use at night hurts sleep quality. Set a phone curfew and use night mode.",
        'low':  " Short or no pre-sleep screen time — this protects your sleep cycle."
    },
    'Phone Use on Waking Up': {
        'high': " Reaching for your phone immediately after waking sets a reactive tone. Try a 15-min screen-free morning routine.",
        'low':  " Starting your day screen-free helps maintain a proactive mindset."
    },
    'Daily Sleep (hrs)': {
        'low':  " You're not getting enough sleep. Aim for 7–8 hours nightly — it directly boosts cognitive performance.",
        'high': " Good sleep duration is one of the strongest predictors of productivity."
    },
    'Daytime Tiredness/Sleepiness': {
        'high': "Feeling tired during the day suggests poor sleep quality or quantity. Review your sleep schedule and reduce late-night screen use.",
        'low':  " Low daytime tiredness — your rest and energy levels are good."
    },
    'Daily Study Hours': {
        'low':  " You study fewer hours per day. Gradually increase focused study time using techniques like Pomodoro (25 min on, 5 min break).",
        'high': " Good daily study duration — maintain consistency for best results."
    },
    'CGPA': {
        'low':  " Your CGPA suggests room for academic improvement. Focus on consistency, attend classes, and seek help early.",
        'high': " Strong CGPA — keep maintaining your academic habits."
    },
    'Focus During Study': {
        'low':  " Difficulty focusing during study is a key concern. Remove phone from your study space and use focus apps like Forest.",
        'high': " Good focus during studies — this is a major driver of productivity."
    },
    'Digital Usage = Main Stress Source': {
        'high': " Your digital usage is causing stress. Take regular digital breaks and practice mindfulness for 10 min daily.",
        'low':  " Digital usage isn't your primary stressor — a positive sign."
    },
    
    'Device Addiction Level': {
        'high': " Signs of device addiction detected. Try a weekly 'digital detox' day and track usage via screen-time apps.",
        'low':  " Low device addiction — you have healthy control over your device use."
    },
    'Stress After Long Screen Time': {
        'high': " Long screen sessions cause you stress. Follow the 20-20-20 rule: every 20 min, look 20 ft away for 20 sec.",
        'low':  " Screen time doesn't significantly stress you out."
    },
    'Online Time Affects Mood': {
        'high': " Your mood is affected by time spent online. Be intentional — curate positive content and limit doom-scrolling.",
        'low':  " Your mood is not significantly impacted by online time."
    },
    'Phone-Separation Anxiety': {
        'high': " High anxiety when away from your phone is a concern. Practice intentional phone-free periods to build comfort.",
        'low':  " You're comfortable without your phone — a sign of healthy digital boundaries."
    },
   
}

# --- STEP 3: MAPPING LOGIC ---
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

# --- STEP 4: FUNCTIONS ---
def get_factor_direction(feature_label, feature_value, shap_val):
    # Logic to decide if a value counts as 'high' or 'low' for the suggestions
    negative_high_features = {'Daily Smartphone Usage (hrs)', 'Daily Social Media Time (hrs)', 'Pre-Sleep Phone Duration (hrs)', 'Device Addiction Level'}
    if feature_label in negative_high_features:
        return 'high' if feature_value >= 2 else 'low'
    return 'low' if shap_val < 0 else 'high'

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

# --- STEP 5: UI ---
st.title("📊 Personal Productivity Consultant")
st.sidebar.header("Settings")
model_choice = st.sidebar.selectbox("Model", ("Random Forest", "Logistic Regression", "Decision Tree"))

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("What is your age?", 10, 100, 20)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    usage = st.selectbox("Daily Smartphone Usage", list(smartphone_map.keys()))
    notif = st.selectbox("How often do you check notifications?", list(notification_map.keys()))
    sleep_before = st.selectbox("Do you use phone before sleeping?", ["Yes", "No"])
    sleep_time = st.selectbox("Hours before sleep usage", list(sleep_map_time.keys()))
    wake_up = st.selectbox("Use device immediately after waking up?", ["Yes", "No"])
    daily_sleep = st.selectbox("Total sleep hours daily?", list(hour_map.keys()))

with col2:
    tired = st.selectbox("Feel tired/sleepy during the day?", list(tired_map.keys()))
    study_hrs = st.selectbox("Daily Study Hours?", list(study_map.keys()))
    cgpa_val = st.selectbox("What is your CGPA?", list(cgpa_map.keys()))
    focus_val = st.selectbox("Able to focus during study?", ["yes", "no"])
    stress_val = st.selectbox("Digital usage is main source of stress?", ["yes", "no"])
    addict_val = st.selectbox("Feel addicted to digital devices?", list(addicted_map.keys()))
    scr_stress = st.selectbox("Stressed after long screen time?", list(stressed_map.keys()))
    mood_val = st.selectbox("Spending time online affects mood?", list(spending_map.keys()))
    anxious_val = st.selectbox("Anxious without phone/internet?", list(anxious_map.keys()))

# --- STEP 6: PREDICTION ---
if st.button("Generate Detailed Report"):
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
        'what is your cgpa?': cgpa_map[cgpa_val],
        'Are you able to focus while study sessions?': yes_no_1map[focus_val],
        'is your digital usage is main source of your stress?': yes_no_2map[stress_val],
        'Do you feel addicted to digital devices?': addicted_map[addict_val],
        'Do you feel stressed after long hours of screen time?': stressed_map[scr_stress],
        'Does spending a lot of time online affect your mood?': spending_map[mood_val],
        'Do you feel anxious when you cannot access your phone or internet?': anxious_map[anxious_val]
    }
    
    input_df = pd.DataFrame([data])
    sel_model = rf if model_choice == "Random Forest" else (lr if model_choice == "Logistic Regression" else dt)
    
    with st.spinner('Calculating impact factors...'):
        shap_vals, tgt_cls, probas = manual_shap_single(sel_model, input_df, X_train_bg)
    
    st.markdown("---")
    st.header(f"{CLASS_EMOJIS[tgt_cls]} Result: {CLASS_NAMES[tgt_cls]}")
    st.write(f"Confidence: {probas[tgt_cls]*100:.1f}%")

    # Factors & Suggestions
    st.subheader("💡 Key Insights & Suggestions")
    local_df = pd.DataFrame({'Feature': input_df.columns, 'SHAP': shap_vals, 'Value': input_df.values.flatten()})
    local_df['Label'] = local_df['Feature'].map(lambda x: FEATURE_LABELS.get(x, x))
    
    # Sort by absolute impact
    top_factors = local_df.reindex(local_df.SHAP.abs().sort_values(ascending=False).index).head(5)

    for _, row in top_factors.iterrows():
        label = row['Label']
        shap_v = row['SHAP']
        direction = get_factor_direction(label, row['Value'], shap_v)
        
        with st.expander(f"{'✅' if shap_v > 0 else '⚠️'} {label}"):
            st.write(f"**Impact:** {'Positive' if shap_v > 0 else 'Negative'}")
            if label in SUGGESTIONS:
                st.info(SUGGESTIONS[label].get(direction, "Keep optimizing this habit."))

    if tgt_cls == 0:
        st.error("You are predicted to have LOW PRODUCTIVITY.Your digital usage patterns are significantly impactingyour academic performance and wellbeing. Start with smallchanges: reduce phone time by 1 hr/day, sleep by 10pm,and keep phones away during study. Consistency is key..")
    elif tgt_cls == 1:
        st.warning("You are predicted to be MEDIUM PRODUCTIVE.There is room for improvement in your digital habits. Address the key factors highlighted above — especiallyscreen time, sleep, and focus during study sessions.")
    else:
        st.success("You are predicted to be HIGHLY PRODUCTIVE. Your digital habits and study patterns are well-balanced.Keep maintaining your current routine and focus habits. Focus on sustaining sleep quality and study consistency.")
