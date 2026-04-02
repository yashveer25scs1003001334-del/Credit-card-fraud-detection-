# Credit-card-fraud-detection-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import roc_curve, auc
from tensorflow.keras.callbacks import EarlyStopping

# 1. Load data
df = pd.read_csv('creditcard.csv')
print(f"Shape of dataset: {df.shape}")
print(df.head())

# 2. Basic EDA
print("Checking for missing values:", df.isnull().sum().any())
print("Missing values per column:\n", df.isnull().sum())
print("Class distribution:\n", df['Class'].value_counts())

# 3. Feature & Target separation
X = df.drop(['Class', 'Time'], axis=1)
y = df['Class']

# 4. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 5. Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define function to print metrics
def print_metrics(name, y_true, y_pred, y_proba):
    print(f"\n{name} Metrics")
    print("F1 Score:", f1_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("ROC AUC:", roc_auc_score(y_true, y_proba))
    print("Classification Report:\n", classification_report(y_true, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

# 6. Logistic Regression
lr = LogisticRegression(max_iter=500, random_state=42)
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
print_metrics("Logistic Regression", y_test, y_pred_lr, y_prob_lr)

# 7. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_scaled, y_train)

y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
print_metrics("Random Forest", y_test, y_pred_rf, y_prob_rf)

# 8. Neural Network (Keras)
def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_dim=input_dim),
        Dropout(0.5),
        Dense(32, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid'),
    ])
    model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Add early stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model = create_model(X_train_scaled.shape[1])
history = model.fit(
    X_train_scaled, y_train, epochs=10, batch_size=64, verbose=1,
    validation_split=0.1, callbacks=[early_stopping]
)

# Predict on test set
y_prob_nn = model.predict(X_test_scaled).flatten()
y_pred_nn = (y_prob_nn > 0.5).astype(int)
print_metrics("Neural Network", y_test, y_pred_nn, y_prob_nn)

# 9. Plot ROC curves for comparison
plt.figure(figsize=(8,6))

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_prob_nn)

plt.plot(fpr_lr, tpr_lr, label='Logistic Regression (AUC = %.4f)' % auc(fpr_lr, tpr_lr))
plt.plot(fpr_rf, tpr_rf, label='Random Forest (AUC = %.4f)' % auc(fpr_rf, tpr_rf))
plt.plot(fpr_nn, tpr_nn, label='Neural Network (AUC = %.4f)' % auc(fpr_nn, tpr_nn))
plt.plot([0,1],[0,1],'k--')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()
