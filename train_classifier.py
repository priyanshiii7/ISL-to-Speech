import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb

data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels
)

model = xgb.XGBClassifier(n_estimators=300, max_depth=15, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss')
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_predict, y_test)

print(f'🎯 XGBoost Accuracy: {score * 100:.2f}%')

with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)

print('✅ Model saved successfully.')
