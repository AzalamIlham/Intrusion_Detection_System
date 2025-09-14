# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

# -------------------------------
# 1. Lecture du fichier CSV
# -------------------------------
file_path = "KDDTrain+.csv"  # trafic réseau et des attaques.

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
# 2. Nettoyage
# -------------------------------
df = df.drop(columns=['difficulty'])

# -------------------------------
# 3. Encodage des variables catégorielles
# -------------------------------
df_encoded = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])

# -------------------------------
# 4. Regrouper les attaques en 5 classes
# -------------------------------
attack_mapping = {
    'neptune':'DoS', 'smurf':'DoS', 'pod':'DoS', 'teardrop':'DoS', 'back':'DoS',
    'land':'DoS',
    'satan':'Probe', 'ipsweep':'Probe', 'nmap':'Probe', 'portsweep':'Probe',
    'guess_passwd':'R2L', 'ftp_write':'R2L', 'warezclient':'R2L', 'warezmaster':'R2L',
    'imap':'R2L', 'phf':'R2L', 'multihop':'R2L',
    'buffer_overflow':'U2R', 'loadmodule':'U2R', 'rootkit':'U2R', 'perl':'U2R', 'spy':'U2R',
    'normal':'Normal'
}

df_encoded['label'] = df_encoded['label'].apply(lambda x: attack_mapping[x])

# -------------------------------
# 5. Séparer features et labels
# -------------------------------
X = df_encoded.drop(columns=['label'])
y = df_encoded['label']

# -------------------------------
# 6. Sur-échantillonnage SMOTE pour équilibrer les classes
# -------------------------------
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# -------------------------------
# 7. Standardiser les features
# -------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_res)

# -------------------------------
# 8. Séparer train/test
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_res, test_size=0.2, random_state=42, stratify=y_res
)

# -------------------------------
# 9. Entraîner un modèle Random Forest
# -------------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# -------------------------------
# 10. Prédictions et évaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\n--- Classification Report ---\n")
print(classification_report(y_test, y_pred))

# ===============================
# 🔥  Visualisations
# ===============================

# 1️⃣ Matrice de confusion en heatmap
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Prédictions")
plt.ylabel("Réelles")
plt.title("Matrice de confusion - Random Forest")
plt.show()

# 2️⃣ Importance des features
importances = model.feature_importances_
indices = importances.argsort()[-15:][::-1]  # Top 15
plt.figure(figsize=(10,6))
plt.bar(range(len(indices)), importances[indices], align="center")
plt.xticks(range(len(indices)), [X.columns[i] for i in indices], rotation=90)
plt.title("Top 15 des features les plus importantes")
plt.savefig("confusion_matrix.png")
plt.show()
