import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Function to load data
def load_data():
    df = pd.read_csv('smoking_health_data_final.csv')
    return df

# Function to preprocess data
def preprocess_data(df):
    df['cigs_per_day'].fillna(df['cigs_per_day'].mean(), inplace=True)
    df['chol'].fillna(df['chol'].ffill(), inplace=True)
    df[['high_bp', 'low_bp']] = df['blood_pressure'].str.split('/', expand=True)
    df['high_bp'] = pd.to_numeric(df['high_bp'])
    df['low_bp'] = pd.to_numeric(df['low_bp'])
    heart_disease_criteria = (df['heart_rate'] > 100) | (df['chol'] > 200)
    df['has_heart_disease'] = heart_disease_criteria.astype(int)

    # Store original min and max for later use
    original_ranges = {
        'heart_rate': (df['heart_rate'].min(), df['heart_rate'].max()),
        'cigs_per_day': (df['cigs_per_day'].min(), df['cigs_per_day'].max()),
        'chol': (df['chol'].min(), df['chol'].max()),
        'high_bp': (df['high_bp'].min(), df['high_bp'].max()),
        'low_bp': (df['low_bp'].min(), df['low_bp'].max())
    }

    # Standardize numerical columns
    scaler = StandardScaler()
    numerical_cols = ['heart_rate', 'cigs_per_day', 'chol', 'high_bp', 'low_bp']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=['sex', 'current_smoker'])

    # Ensure all expected columns are present
    for col in ['sex_female', 'sex_male', 'current_smoker_no', 'current_smoker_yes']:
        if col not in df.columns:
            df[col] = 0

    # Convert columns to appropriate data types
    df['sex_female'] = df['sex_female'].astype(int)
    df['sex_male'] = df['sex_male'].astype(int)
    df['current_smoker_no'] = df['current_smoker_no'].astype(int)
    df['current_smoker_yes'] = df['current_smoker_yes'].astype(int)

    return df, original_ranges, scaler

# Function to train model
def train_model(df):
    X = df.drop(columns=['blood_pressure', 'has_heart_disease'])
    y = df['has_heart_disease']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000, solver='liblinear')
    model.fit(X_train, y_train)

    return model, X_test, y_test, X_train.columns

# Main function
def main():
    st.title("Heart Disease Prediction")

    # Load data
    df = load_data()

    # Preprocess data
    df, original_ranges, scaler = preprocess_data(df)

    # Train model
    model, X_test, y_test, feature_names = train_model(df)

    # Reverse standardization to get the original input limits
    def reverse_standardization(scaled_value, mean, std):
        return scaled_value * std + mean

    heart_rate_min, heart_rate_max = original_ranges['heart_rate']
    heart_rate_mean, heart_rate_std = scaler.mean_[0], scaler.scale_[0]

    cigs_per_day_min, cigs_per_day_max = original_ranges['cigs_per_day']
    cigs_per_day_mean, cigs_per_day_std = scaler.mean_[1], scaler.scale_[1]

    chol_min, chol_max = original_ranges['chol']
    chol_mean, chol_std = scaler.mean_[2], scaler.scale_[2]

    high_bp_min, high_bp_max = original_ranges['high_bp']
    high_bp_mean, high_bp_std = scaler.mean_[3], scaler.scale_[3]

    low_bp_min, low_bp_max = original_ranges['low_bp']
    low_bp_mean, low_bp_std = scaler.mean_[4], scaler.scale_[4]

    # Display user number input
    st.sidebar.header("User Input")
    heart_rate = st.sidebar.number_input("Heart Rate", float(heart_rate_min), float(heart_rate_max), float(heart_rate_mean))
    cigs_per_day = st.sidebar.number_input("Cigarettes per Day", float(cigs_per_day_min), float(cigs_per_day_max), float(cigs_per_day_mean))
    chol = st.sidebar.number_input("Cholesterol Level", float(chol_min), float(chol_max), float(chol_mean))
    high_bp = st.sidebar.number_input("High Blood Pressure", float(high_bp_min), float(high_bp_max), float(high_bp_mean))
    low_bp = st.sidebar.number_input("Low Blood Pressure", float(low_bp_min), float(low_bp_max), float(low_bp_mean))
    sex_female = st.sidebar.radio("Sex (Female)", [0, 1], index=0)
    current_smoker_yes = st.sidebar.radio("Current Smoker (Yes)", [0, 1], index=0)
    age = st.sidebar.number_input("Age", 0, 120, 0)

    # Standardize user inputs before prediction
    user_data = pd.DataFrame({
        'heart_rate': [(heart_rate - heart_rate_mean) / heart_rate_std],
        'cigs_per_day': [(cigs_per_day - cigs_per_day_mean) / cigs_per_day_std],
        'chol': [(chol - chol_mean) / chol_std],
        'high_bp': [(high_bp - high_bp_mean) / high_bp_std],
        'low_bp': [(low_bp - low_bp_mean) / low_bp_std],
        'sex_female': [sex_female],
        'sex_male': [1 - sex_female],
        'current_smoker_yes': [current_smoker_yes],
        'current_smoker_no': [1 - current_smoker_yes],
        'age': [age]
    })

    # Reorder user_data columns to match the training data
    user_data = user_data[feature_names]

    # Make prediction
    if st.sidebar.button("Predict"):
        prediction = model.predict(user_data)
        if prediction[0] == 1:
            st.success("The individual may have heart disease.")
        else:
            st.success("The individual may not have heart disease.")

    # Display model accuracy
    accuracy = accuracy_score(y_test, model.predict(X_test))
    st.write("Model Accuracy:", accuracy)

if __name__ == '__main__':
    main()
