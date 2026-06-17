import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

data_dict = pickle.load(open('./data.pickle', 'rb'))
data   = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# encode 'a','b','c'... → 0,1,2...
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

x_train, x_test, y_train, y_test = train_test_split(
    data, labels_encoded, test_size=0.2, shuffle=True, stratify=labels_encoded
)

model = xgb.XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.1, eval_metric='mlogloss')
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)
print(f'XGBoost Accuracy: {score * 100:.2f}%')

# save model AND label encoder so inference can decode predictions
with open('model.p', 'wb') as f:
    pickle.dump({'model': model, 'encoder': le}, f)

print('Model saved to model.p')