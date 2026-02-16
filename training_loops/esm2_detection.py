import pandas as pd 
import numpy as np 
import joblib
import os
import ast
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

df = pd.read_csv("gold_with_esm2.csv")
df.head()


y = df["is_bla"].astype(int).values

#turn embeddings into input features
X = np.vstack(df["ESM-2_embedding"].apply(lambda s: np.array(ast.literal_eval(s), dtype=np.float32)).values)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, random_state=42, stratify=y, test_size=0.1
)

#build LogReg model
clf = Pipeline([
    ("scaler", StandardScaler(with_mean=True)),
    ("lr", LogisticRegression(max_iter=5000, class_weight="balanced", n_jobs=-1)
     )
])

clf.fit(X_train, y_train)

# Probabilities
p = clf.predict_proba(X_val)[:, 1]

# Default threshold (0.5)
y_pred = (p >= 0.5).astype(int)

print("AUROC:   ", roc_auc_score(y_val, p))
print("AUPRC:   ", average_precision_score(y_val, p))
print("Accuracy:", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred, zero_division=0))
print("Recall:   ", recall_score(y_val, y_pred))


# save model
OUT_PATH = "best_bl_lr_gold.joblib"   # or full path if you want

joblib.dump(clf, OUT_PATH)

print("Saved LR model to:", os.path.abspath(OUT_PATH))