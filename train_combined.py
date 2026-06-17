import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import os

print("Loading data...")
all_data, all_labels = [], []

letter_dict = pickle.load(open("data.pickle", "rb"))
letter_data = np.array(letter_dict["data"])
letter_labels = np.array(letter_dict["labels"])
letter_data_42 = letter_data[:, :42]
all_data.extend(letter_data_42.tolist())
all_labels.extend(letter_labels.tolist())
print(f"Letters: {len(letter_labels)} samples")

if os.path.exists("data_words.pickle"):
    word_dict = pickle.load(open("data_words.pickle", "rb"))
    word_data = np.array(word_dict["data"])
    word_labels = np.array(word_dict["labels"])
    all_data.extend(word_data.tolist())
    all_labels.extend(word_labels.tolist())
    print(f"Words: {len(word_labels)} samples, classes: {sorted(set(word_labels))}")

all_data = np.array(all_data)
all_labels = np.array(all_labels)
print(f"Total: {len(all_data)} samples, {len(set(all_labels))} classes, {all_data.shape[1]} features")

le = LabelEncoder()
labels_encoded = le.fit_transform(all_labels)

x_train, x_test, y_train, y_test = train_test_split(
    all_data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded, random_state=42)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(x_train, y_train)
print(f"RF accuracy: {accuracy_score(y_test, rf.predict(x_test))*100:.2f}%")

print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=12, learning_rate=0.1, eval_metric='mlogloss', n_jobs=-1, random_state=42)
xgb_model.fit(x_train, y_train)
print(f"XGB accuracy: {accuracy_score(y_test, xgb_model.predict(x_test))*100:.2f}%")

print("Training ensemble...")
ensemble = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb_model)], voting="soft")
ensemble.fit(x_train, y_train)
acc = accuracy_score(y_test, ensemble.predict(x_test))
print(f"Ensemble accuracy: {acc*100:.2f}%")

print("\n── Per-class report ──")
print(classification_report(y_test, ensemble.predict(x_test), target_names=le.classes_, zero_division=0))

with open("model_combined.p", "wb") as f:
    pickle.dump({"model": ensemble, "encoder": le, "feature_size": 42, "classes": list(le.classes_)}, f)

print(f"✅ Saved model_combined.p")
print(f"Classes: {list(le.classes_)}")