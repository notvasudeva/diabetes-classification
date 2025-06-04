import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

#Loading historical dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

#Simulate diabetes type
df['DiabetesType'] = df.apply(
    lambda row: 1 if (row['Age'] < 30 and row['Insulin'] < 100 and row['Outcome'] == 1) else (
        2 if row['Outcome'] == 1 else 0), axis=1
)

#Feature preparation
X = df.drop(['Outcome', 'DiabetesType'], axis=1)
y_diabetes = df['Outcome']

#Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#Training KNN for diabetes detection
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_scaled, y_diabetes, test_size=0.2, random_state=42)
knn_diabetes = KNeighborsClassifier(n_neighbors=5)
knn_diabetes.fit(X_train_d, y_train_d)

#Training KNN for diabetes type
df_diabetic = df[df['Outcome'] == 1]
X_type = df_diabetic.drop(['Outcome', 'DiabetesType'], axis=1)
y_type = df_diabetic['DiabetesType']
X_type_scaled = scaler.transform(X_type)
X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_type_scaled, y_type, test_size=0.2, random_state=42)
knn_type = KNeighborsClassifier(n_neighbors=3)
knn_type.fit(X_train_t, y_train_t)

#Accuracy scores
accuracy_diabetes = knn_diabetes.score(X_test_d, y_test_d)
accuracy_type = knn_type.score(X_test_t, y_test_t)
print(f"Diabetes Detection Accuracy: {accuracy_diabetes:.2f}")
print(f"Diabetes Type Classification Accuracy: {accuracy_type:.2f}")

#Input from user
def get_user_input():
    print("Please enter the following medical values:\n")
    try:
        inputs = [
            float(input("Pregnancies: ")),
            float(input("Glucose Level: ")),
            float(input("Blood Pressure: ")),
            float(input("Skin Thickness: ")),
            float(input("Insulin Level: ")),
            float(input("BMI: ")),
            float(input("Diabetes Pedigree Function (DPF): ")),
            float(input("Age: "))
        ]
        return inputs
    except ValueError:
        print("Invalid input: Please enter only numeric values.")
        return None

#Full classification logic
def classify_patient(patient_input):
    patient_scaled = scaler.transform([patient_input])
    #Step 1:Diabetes check
    is_diabetic = knn_diabetes.predict(patient_scaled)[0]
    if is_diabetic == 0:
        return "You are NOT diabetic."
    #Step 2:Diabetes type
    diabetes_type = knn_type.predict(patient_scaled)[0]
    if diabetes_type == 1:
        return "You HAVE diabetes — Type 1."
    elif diabetes_type == 2:
        return "You HAVE diabetes — Type 2."
    else:
        return "You have diabetes but type classification is uncertain."

#Run the program
if __name__ == "__main__":
    user_input = get_user_input()
    if user_input:
        result = classify_patient(user_input)
        print("\nResult:", result)