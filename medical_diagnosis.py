import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_excel(r"/content/TestData.xlsx")

X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

from sklearn.preprocessing import StandardScaler
X_features = data.drop(columns=['id_patient', 'observations', 'diagnostic'], errors='ignore')
X_encoded = pd.get_dummies(X_features, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=0
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X_train_scaled, y_train)

y_pred = classifier.predict(X_test_scaled)
print("Accuracy:", classifier.score(X_test_scaled, y_test))
y_pred_encoded = pd.Categorical(y_pred).codes

scatter = plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_pred_encoded, cmap='viridis')
plt.xlabel("Scaled Feature 1 (Age)")
plt.ylabel("Scaled Feature 2 (Years of Exposure)")
plt.title("Medical Disease Diagnosis Prediction")
plt.legend()
plt.show()

