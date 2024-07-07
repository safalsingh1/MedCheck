import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the dataset
file_path = 'Disease_symptom_and_patient_profile_dataset.csv'
data = pd.read_csv(file_path)

# Encode categorical variables
label_encoders = {}
for column in ['Disease', 'Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Gender', 'Blood Pressure', 'Cholesterol Level', 'Outcome Variable']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Split the data into features and target
X = data.drop(columns=['Outcome Variable'])
y = data['Outcome Variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature variables
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model, label encoders, and scaler
with open('model.pkl', 'wb') as file:
    pickle.dump((model, label_encoders, scaler), file)

print("Model, label encoders, and scaler saved to 'model.pkl'.")
