import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pickle


data = pd.read_csv('data/gesture-data.csv')

gesture_map = {
    'stop': 0,
    'thumbs-down': 1,
    'thumbs-up': 2,
    'excuse me': 3
}

X = data.copy()
X['gesture'] = X['gesture'].replace(gesture_map)
y = X.pop('gesture')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

xgb = XGBClassifier()
xgb.fit(X_train, y_train)

pred = xgb.predict(X_test)
print(f'Training accuracy: {accuracy_score(pred, y_test) * 100:.2f}%')

print('Saving model... ', end='')
with open('models/classifier.pkl', 'wb') as f:
    pickle.dump(xgb, f)
print('done.')