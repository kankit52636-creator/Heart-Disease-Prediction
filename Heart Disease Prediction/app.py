import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# ==========================================
# PAGE SETUP
# ==========================================
st.set_page_config(page_title="Heart Disease Predictor", layout="wide")
st.title("🫀 Heart Disease Prediction Dashboard")
st.write("Enter patient vitals in the sidebar to generate a real-time risk assessment.")

# ==========================================
# MODEL TRAINING & CACHING
# ==========================================
# We cache this so the model doesn't retrain every time you click a button
@st.cache_resource
def build_and_train_model():
    file_path = "heart.csv" # Update this to "/content/heart.csv" if running in Colab
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Could not find {file_path}. Please check your file path.")
        st.stop()
        
    df = df.drop_duplicates()

    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop('target', axis=1)
    y = df_encoded['target']
    feature_columns = X.columns

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train[continuous_cols] = scaler.fit_transform(X_train[continuous_cols])
    X_test[continuous_cols] = scaler.transform(X_test[continuous_cols])

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    return rf_model, scaler, feature_columns, accuracy, X_test, y_test, y_pred, y_prob

# Load our cached model and data
rf_model, scaler, feature_columns, accuracy, X_test, y_test, y_pred, y_prob = build_and_train_model()

# ==========================================
# USER INPUT SIDEBAR
# ==========================================
st.sidebar.header("Patient Vitals")

# We use more appropriate input types here compared to Tkinter
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=55)
sex = st.sidebar.selectbox("Sex", options=[1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
cp = st.sidebar.selectbox("Chest Pain Type (0-3)", options=[0, 1, 2, 3])
trestbps = st.sidebar.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=130)
chol = st.sidebar.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", options=[0, 1], format_func=lambda x: "True (1)" if x == 1 else "False (0)")
restecg = st.sidebar.selectbox("Resting ECG (0-2)", options=[0, 1, 2])
thalach = st.sidebar.number_input("Maximum Heart Rate", min_value=60, max_value=250, value=150)
exang = st.sidebar.selectbox("Exercise Angina", options=[0, 1], format_func=lambda x: "Yes (1)" if x == 1 else "No (0)")
oldpeak = st.sidebar.number_input("ST Depression", min_value=0.0, max_value=10.0, value=1.5, step=0.1)
slope = st.sidebar.selectbox("Slope (0-2)", options=[0, 1, 2])
ca = st.sidebar.selectbox("Major Vessels (0-4)", options=[0, 1, 2, 3, 4])
thal = st.sidebar.selectbox("Thal (0-3)", options=[0, 1, 2, 3])

# ==========================================
# PREDICTION LOGIC
# ==========================================
if st.sidebar.button("Analyze Vitals", type="primary"):
    
    user_data = pd.DataFrame({
        'age': [age], 'sex': [sex], 'cp': [cp], 'trestbps': [trestbps],
        'chol': [chol], 'fbs': [fbs], 'restecg': [restecg], 'thalach': [thalach],
        'exang': [exang], 'oldpeak': [oldpeak], 'slope': [slope],
        'ca': [ca], 'thal': [thal]
    })

    # Preprocess identically to training
    categorical_cols = ['cp', 'restecg', 'slope', 'thal']
    continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    user_encoded = pd.get_dummies(user_data, columns=categorical_cols, drop_first=True)
    user_encoded = user_encoded.reindex(columns=feature_columns, fill_value=0)
    user_encoded[continuous_cols] = scaler.transform(user_encoded[continuous_cols])

    prediction = rf_model.predict(user_encoded)[0]
    probability = rf_model.predict_proba(user_encoded)[0]
    
    votes_healthy = int(round(probability[0] * 100))
    votes_sick = int(round(probability[1] * 100))

    st.subheader("Diagnostic Results")
    
    # Display results cleanly
    col1, col2, col3 = st.columns(3)
    col1.metric("Overall Model Accuracy", f"{accuracy * 100:.1f}%")
    
    if prediction == 1:
        st.error("### ⚠️ HIGH RISK OF HEART DISEASE DETECTED")
        col2.metric("Prediction Confidence", f"{probability[1] * 100:.1f}%")
        st.write(f"**How the algorithm voted:** {votes_sick} trees voted Sick, {votes_healthy} voted Healthy.")
    else:
        st.success("### ✅ LOW RISK DETECTED")
        col2.metric("Prediction Confidence", f"{probability[0] * 100:.1f}%")
        st.write(f"**How the algorithm voted:** {votes_healthy} trees voted Healthy, {votes_sick} voted Sick.")
        
    st.caption("Note: This is a statistical prediction based on training data and not official medical advice.")
    st.divider()

# ==========================================
# MODEL METRICS & VISUALIZATIONS
# ==========================================
st.subheader("Model Performance Metrics")
tab1, tab2, tab3 = st.tabs(["Confusion Matrix", "Feature Importance", "ROC Curve"])

with tab1:
    st.write("This shows how often the model confused a healthy patient for a sick one, and vice versa.")
    fig, ax = plt.subplots(figsize=(6, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
                xticklabels=['Predicted Healthy', 'Predicted Sick'],
                yticklabels=['Actual Healthy', 'Actual Sick'], ax=ax)
    st.pyplot(fig)

with tab2:
    st.write("This highlights which vitals heavily influenced the model's decision making process.")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_columns[i] for i in indices]
    sorted_importances = [importances[i] for i in indices]

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    sns.barplot(x=sorted_importances, y=sorted_features, ax=ax2, palette='viridis')
    ax2.set_xlabel('Relative Importance (Gini Index)')
    st.pyplot(fig2)

with tab3:
    st.write("A visual measure of the model's ability to distinguish between the two classes.")
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax3.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc="lower right")
    st.pyplot(fig3)
