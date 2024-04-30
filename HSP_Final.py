import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Read the dataset
df = pd.read_csv("C:\\Users\\admin\\Desktop\\healthcare-dataset-stroke-data.csv")

# Handle missing values by replacing with mean
m = df['bmi'].mean()
df['bmi'] = df['bmi'].fillna(m)

# Encode categorical variables
label_encoders = {}
for column in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare data for modeling
x = df.iloc[:,1:10]
y = df['stroke']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=51)

# Train SVM model
lr=LogisticRegression()
lr.fit(x_train, y_train)

# Evaluate model
y_pred = lr.predict(x_test)
print("Accuracy score:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



# Take user inputs
P_gender = input("Enter gender (male/female/other): ").lower()  # male/female/other

# Encoding gender
if P_gender == 'female':
    Pe_gender = 0
elif P_gender == 'male':
    Pe_gender = 1
else:
    Pe_gender = 2  # Assuming 'other' is encoded as 2

P_age = int(input("Enter patient's age: "))
P_Avg_glu = float(input("Enter Avg Glucose levels: "))
P_Bmi = float(input("Enter BMI: "))

P_smoke_status = input("Enter Smoking status (Currently Smokes/Formerly Smoked/Non-smoker/unknown): ").lower()

# Encoding smoking status
if P_smoke_status == 'currently smokes':
    Pe_smoke_status = 0
elif P_smoke_status == 'formerly smoked':
    Pe_smoke_status = 1
elif P_smoke_status == 'never smoked':
    Pe_smoke_status = 2
else:
    Pe_smoke_status = 3  # Assuming 'unknown' is encoded as 3

P_Work_Type = input("Work Type (Govt_job/Never_worked/Private/Self-employed/Children): ").lower()

# Encoding work type
if P_Work_Type == 'govt_job':
    Pe_Work_Type = 0
elif P_Work_Type == 'never_worked':
    Pe_Work_Type = 1
elif P_Work_Type == 'private':
    Pe_Work_Type = 2
elif P_Work_Type == 'self-employed':
    Pe_Work_Type = 3
else:
    Pe_Work_Type = 4  # Assuming 'children' is encoded as 4

P_Residence_Type = input("Residence (Urban/Rural): ").lower()  # Urban/rural

# Encoding residence type
if P_Residence_Type == 'rural':
   Pe_Residence_Type = 0
else:
   Pe_Residence_Type = 1

# Validate and encode boolean inputs
P_ever_married = input("Enter 1 if patient was ever married (0 otherwise): ")
P_ever_married = bool(int(P_ever_married))  # Convert input to boolean

P_stress_levels = float(input("Stress Level Scores: "))

P_hypertension = input("Enter 1 if patient has hypertension (0 otherwise): ")
P_hypertension = bool(int(P_hypertension))  # Convert input to boolean

P_heart_disease = input("Enter 1 if patient has heart disease (0 otherwise): ")
P_heart_disease = bool(int(P_heart_disease))  # Convert input to boolean

# Create new data instance
new_data = [[Pe_gender, P_age, P_hypertension, P_heart_disease, P_ever_married, Pe_Work_Type, Pe_Residence_Type, P_Bmi, Pe_smoke_status]]

# Predict using SVM model
predicted_class =lr.predict(new_data)

# Output prediction result
if predicted_class == 1:
    print("High Chances of Stroke")
else:
    print("Low chances of stroke")
