# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------------
# 1. Lecture du fichier CSV
# -------------------------------
file_path = "KDDTrain+.csv" 

# Liste des colonnes (41 features + label + difficulty)
columns = [
    "duration","protocol_type","service","flag","src_bytes","dst_bytes","land",
    "wrong_fragment","urgent","hot","num_failed_logins","logged_in","num_compromised",
    "root_shell","su_attempted","num_root","num_file_creations","num_shells",
    "num_access_files","num_outbound_cmds","is_host_login","is_guest_login",
    "count","srv_count","serror_rate","srv_serror_rate","rerror_rate","srv_rerror_rate",
    "same_srv_rate","diff_srv_rate","srv_diff_host_rate","dst_host_count",
    "dst_host_srv_count","dst_host_same_srv_rate","dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate","dst_host_srv_diff_host_rate",
    "dst_host_serror_rate","dst_host_srv_serror_rate","dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label","difficulty"
]

df = pd.read_csv(file_path, names=columns)

# -------------------------------
# 2.Nettoyage et préparation
# -------------------------------
df = df.drop(columns=['difficulty'])
print("Shape:", df.shape)
print("Colonnes:", df.columns.tolist())
print("Labels uniques:", df['label'].unique())

# -------------------------------
# 3. Encoder les colonnes catégorielles
# -------------------------------
df_encoded = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# -------------------------------
# 4. Transformer les labels en binaire
# -------------------------------
# normal -> normal, toutes les attaques -> attack
df_encoded['label'] = df_encoded['label'].apply(lambda x: 'normal' if x=='normal' else 'attack')

# -------------------------------
# 5. Séparer features et labels
# -------------------------------
X = df_encoded.drop(columns=['label'])
y = df_encoded['label']

# -------------------------------
# 6. Standardiser les features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -------------------------------
# 7. Séparer train et test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# -------------------------------
# 8. Entraîner un modèle Random Forest
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------------
# 9. Prédictions et évaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---\n")
print(confusion_matrix(y_test, y_pred))
