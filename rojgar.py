import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load and clean dataset
df = pd.read_csv("job.csv")
df.drop_duplicates(inplace=True)
df.dropna(subset=['Job Title', 'Automation Risk (%)', 'Experience Required (Years)'], inplace=True)

# Create Risk Category for classification
bins = [0, 30, 70, 100]
labels = ['Low Risk', 'Medium Risk', 'High Risk']
df['Risk Category'] = pd.cut(df['Automation Risk (%)'], bins=bins, labels=labels, include_lowest=True)

# Select fewer input features for simplicity
selected_features = ['Industry', 'Experience Required (Years)', 'Remote Work Ratio (%)']

# Encode categorical column
le_industry = LabelEncoder()
df['Industry Encoded'] = le_industry.fit_transform(df['Industry'])

# Prepare final feature matrix X and target y
X = df[['Industry Encoded', 'Experience Required (Years)', 'Remote Work Ratio (%)']]
y = df['Risk Category']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Scale numerical features
scaler = StandardScaler()
X_train[X.select_dtypes(include=np.number).columns] = scaler.fit_transform(X_train[X.select_dtypes(include=np.number).columns])
X_test[X.select_dtypes(include=np.number).columns] = scaler.transform(X_test[X.select_dtypes(include=np.number).columns])

# Train classifier
model = GradientBoostingClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

# === 📈 VISUALIZATIONS SECTION ===

# 1. Confusion Matrix Heatmap
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 2. Classification Report (Bar Plot)
report = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report).transpose().drop(['accuracy', 'macro avg', 'weighted avg'])
report_df[['precision', 'recall', 'f1-score']].plot(kind='bar', figsize=(8, 5))
plt.title("Precision, Recall, F1-score per Class")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()

# 3. Feature Importance Plot
feature_names = ['Industry Encoded', 'Experience (Years)', 'Remote Work Ratio (%)']
importances = model.feature_importances_
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_imp_df.sort_values(by='Importance', ascending=False, inplace=True)

plt.figure(figsize=(6, 4))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

# 4. Accuracy as Bar Chart
accuracy = accuracy_score(y_test, y_pred)
plt.figure(figsize=(4, 4))
plt.bar(['Accuracy'], [accuracy])
plt.ylim(0, 1)
plt.title(f"Model Accuracy: {accuracy*100:.2f}%")
plt.tight_layout()
plt.show()

# === User Input Prediction ===
print("\n--- AI Replaceability Risk Predictor ---")

user_industry = input(f"Industry (options: {', '.join(le_industry.classes_)}): ").strip()
if user_industry not in le_industry.classes_:
    print("Industry not recognized. Using 'Unknown' encoding.")
    user_industry_encoded = 0
else:
    user_industry_encoded = le_industry.transform([user_industry])[0]

try:
    user_exp = float(input("Experience (Years): "))
    user_remote = float(input("Remote Work Ratio (%): "))
except ValueError:
    print("Invalid numeric input. Please enter valid numbers.")
    exit()

X_user = pd.DataFrame([[user_industry_encoded, user_exp, user_remote]],
                      columns=['Industry Encoded', 'Experience Required (Years)', 'Remote Work Ratio (%)'])
X_user[X_user.select_dtypes(include=np.number).columns] = scaler.transform(X_user[X_user.select_dtypes(include=np.number).columns])

prediction = model.predict(X_user)[0]
print(f"\nPredicted AI Automation Risk Category: {prediction}")
