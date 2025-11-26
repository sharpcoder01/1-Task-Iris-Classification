import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("titanic.csv")

# Drop irrelevant columns
df_model = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])

# Encode categorical variables
label_encoders = {}
for column in ["Sex", "Embarked"]:
    le = LabelEncoder()
    df_model[column] = le.fit_transform(df_model[column].astype(str))
    label_encoders[column] = le

# Impute missing values
imputer = SimpleImputer(strategy='mean')
df_model[df_model.columns] = imputer.fit_transform(df_model)

# Split features and target
X = df_model.drop(columns=["Survived"])
y = df_model["Survived"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediction and evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_test, y_pred))
