import streamlit as st
import pandas as pd
import joblib
import numpy as np
import io

# --- 1. ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        lr = joblib.load('logistic_regression.pkl')
        rf = joblib.load('random_forest.pkl')
        dt = joblib.load('decision_tree.pkl')
        X_train = pd.read_csv('train_reference.csv') 
        return lr, rf, dt, X_train
    except Exception as e:
        st.error(f"Asset loading failed: {e}. Ensure .pkl and .csv files are in GitHub.")
        return None, None, None, None

lr, rf, dt, X_train_bg = load_assets()

# --- 2. SESSION STATE MANAGEMENT ---
if 'started' not in st.session_state:
    st.session_state.started = False

def start_analysis():
    st.session_state.started = True

def go_home():
    st.session_state.started = False

# --- 3. DICTIONARIES & MAPPINGS ---
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
    'Daily Smartphone Usage (hrs)': {'high': "Your smartphone usage is high. Try setting daily limits (under 3 hrs).", 'low': "Usage is well-controlled."},
    'Daily Social Media Time (hrs)': {'high': "Excessive social media lowers productivity. Try a 30-min daily cap.", 'low': "Social media time is healthy."},
    'Notification Checking Frequency': {'high': "Frequent checks break focus. Use 'Do Not Disturb' while studying.", 'low': "Infrequent checks — great for deep focus."},
    'Phone Use Before Sleep': {'high': "Screen time disrupts sleep. Stop 60 min before bed.", 'low': "Avoiding pre-sleep phone use supports quality rest."},
    'Pre-Sleep Phone Duration (hrs)': {'high': "Long night usage hurts sleep. Set a phone curfew.", 'low': "Short screen time protects your sleep cycle."},
    'Phone Use on Waking Up': {'high': "Avoid reaching for your phone immediately. Try a screen-free morning routine.", 'low': "Starting screen-free keeps you proactive."},
    'Daily Sleep (hrs)': {'low': "Aim for 7–8 hours; it directly boosts cognitive performance.", 'high': "Good sleep duration is a strong predictor of success."},
    'Daytime Tiredness/Sleepiness': {'high': "Tiredness suggests poor sleep. Review your schedule and night screen use.", 'low': "Energy levels are good."},
    'Daily Study Hours': {'low': "Gradually increase focused study time using the Pomodoro technique.", 'high': "Good study duration — maintain consistency."},
    'CGPA': {'low': "CGPA suggests room for improvement. Focus on consistency.", 'high': "Strong CGPA — keep it up."},
    'Focus During Study': {'low': "Remove your phone from your study space to improve focus.", 'high': "Good focus is a major driver of your productivity."},
    'Digital Usage = Main Stress Source': {'high': "Usage is causing stress. Take regular digital breaks.", 'low': "Digital usage isn't your primary stressor."},
    'Device Addiction Level': {'high': "Signs of addiction detected. Try a weekly 'digital detox'.", 'low': "Healthy control over device use."},
    'Stress After Long Screen Time': {'high': "Follow the 20-20-20 rule to reduce screen stress.", 'low': "Screen time doesn't significantly stress you."},
    'Online Time Affects Mood': {'high': "Curate positive content and limit doom-scrolling.", 'low': "Mood is not significantly impacted by online time."},
    'Phone-Separation Anxiety': {'high': "Practice intentional phone-free periods to reduce anxiety.", 'low': "Healthy digital boundaries detected."}
}

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

# --- 4. CALCULATION FUNCTIONS ---
def get_factor_direction(feature_label, feature_value, shap_val):
    neg_high = {'Daily Smartphone Usage (hrs)', 'Daily Social Media Time (hrs)', 'Pre-Sleep Phone Duration (hrs)'}
    if feature_label in neg_high:
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
            for j in perm[:k+1]: s_with.iloc[:, j] = sample_arr[j]
            pred_with = model.predict_proba(s_with).mean(axis=0)
            shapley[idx] += (pred_with - prev_val)[target_class]
            prev_val = pred_with
    return shapley / n_repeats, target_class, pred_proba

# --- 5. PAGE NAVIGATION ---
if not st.session_state.started:
    # --- LANDING PAGE ---
    st.title("🚀 Digital Usage Productivity Analyzer")
    st.markdown("""
    ### Welcome!
    This application uses AI to analyze how your digital habits (smartphone use, social media, and sleep)
    directly impact your academic focus and productivity.
    
    **How it works:**
    1. Provide your behavioral data via our form.
    2. Our AI models predict your productivity classification.
    3. We generate a technical report showing which factors are boosting or hurting you.
    """)
    st.button("Start My Analysis", on_click=start_analysis, type="primary")

else:
    # --- FORM PAGE ---
    st.title("📝 Behavioral Data Entry")
    st.sidebar.header("Configuration")
    model_choice = st.sidebar.selectbox("Model", ("Random Forest", "Logistic Regression", "Decision Tree"))
    st.sidebar.button("Back to Home", on_click=go_home)

    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("What is your age?", 10, 100, 20)
        gender = st.selectbox("Gender", list(gender_map.keys()))
        usage = st.selectbox("Daily Smartphone Usage", list(smartphone_map.keys()))
        notif = st.selectbox("How often do you check notifications?", list(notification_map.keys()))
        social_media = st.selectbox("Social media time (hours)", list(approx_map.keys()))
        sleep_before = st.selectbox("Use phone before sleeping?", ["Yes", "No"])
        sleep_time = st.selectbox("Hours before sleep usage", list(sleep_map_time.keys()))
        wake_up = st.selectbox("Use device after waking?", ["Yes", "No"])
        daily_sleep = st.selectbox("Total daily sleep?", list(hour_map.keys()))

    with col2:
        tired = st.selectbox("Feel tired during the day?", list(tired_map.keys()))
        study_hrs = st.selectbox("Daily Study Hours?", list(study_map.keys()))
        cgpa_val = st.selectbox("Your CGPA?", list(cgpa_map.keys()))
        focus_val = st.selectbox("Able to focus?", ["yes", "no"])
        stress_val = st.selectbox("Usage is main stress source?", ["yes", "no"])
        addict_val = st.selectbox("Feel addicted?", list(addicted_map.keys()))
        scr_stress = st.selectbox("Stressed after screen time?", list(stressed_map.keys()))
        mood_val = st.selectbox("Online time affects mood?", list(spending_map.keys()))
        anxious_val = st.selectbox("Anxious without phone?", list(anxious_map.keys()))

    if st.button("Generate Detailed Report"):
        data = {
            'What is your age?': age, 'Gender?': gender_map[gender],
            'How many hours per day do you use your smartphone?': smartphone_map[usage],
            'Approximate time spent per day on Social media ?(in hours)': approx_map[social_media],
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
        
        if X_train_bg is not None:
            with st.spinner("Analyzing data..."):
                shap_vals, pred_label, pred_proba = manual_shap_single(sel_model, input_df, X_train_bg)

            # --- GENERATE FORMATTED REPORT ---
            report_output = io.StringIO()
            print("=" * 62, file=report_output)
            print("        PRODUCTIVITY PREDICTION REPORT", file=report_output)
            print("=" * 62, file=report_output)
            print(f"  Predicted Class  : {CLASS_EMOJIS[pred_label]} {CLASS_NAMES[pred_label]}", file=report_output)
            print(f"  Confidence       : {pred_proba[pred_label]*100:.1f}%", file=report_output)
            print(f"  All Probabilities: Low={pred_proba[0]:.3f} | Medium={pred_proba[1]:.3f} | High={pred_proba[2]:.3f}", file=report_output)
            print("-" * 62, file=report_output)
            print("\n KEY CONTRIBUTING FACTORS", file=report_output)
            print("-" * 62, file=report_output)
            print(f"  {'#':<3} {'Factor':<38} {'Impact':>8}  {'Effect'}", file=report_output)
            print(f"  {'-'*3} {'-'*38} {'-'*8}  {'-'*20}", file=report_output)

            local_df = pd.DataFrame({'Feature': input_df.columns, 'Value': input_df.values.flatten(), 'SHAP': shap_vals})
            local_df['Label'] = local_df['Feature'].map(lambda f: FEATURE_LABELS.get(f, f))
            local_df['Abs'] = local_df['SHAP'].abs()
            top = local_df.sort_values('Abs', ascending=False).head(6)

            for i, row in top.reset_index(drop=True).iterrows():
                label = row['Label'][:37]
                shap_v = row['SHAP']
                direction = "▲ Boosts" if shap_v > 0 else "▼ Hurts "
                bar = "+" * min(int(abs(shap_v) * 200), 10) if shap_v > 0 else "-" * min(int(abs(shap_v) * 200), 10)
                print(f"  {i+1:<3} {label:<38} {shap_v:>+8.4f}  {direction} [{bar}]", file=report_output)

            st.code(report_output.getvalue(), language="text")

            # --- SUGGESTIONS ---
            st.markdown("### 💡 Recommendations")
            for _, row in top.iterrows():
                label = row['Label']
                dir_key = get_factor_direction(label, row['Value'], row['SHAP'])
                if label in SUGGESTIONS:
                    st.info(f"**{label}:** {SUGGESTIONS[label].get(dir_key)}")
        else:
            st.error("Cannot compute report. Reference data missing.")
