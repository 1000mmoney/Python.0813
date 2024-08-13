# matplotlib 설치
import matplotlib.pyplot as plt

#pandas 설치
import pandas as pd

# numpy 설치
import numpy as np

# scikit-learn 설치
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# 각 항목 이름 붙이기
colums = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names=colums)

# 바꾸기 할 항목 범위 정하기
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# 소수점 아래로 바꾸기 1, 데이터 전처리 : Min-Max 스케일링
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)


# 모델 선택 및 학습
model =LogisticRegression()

fold = KFold(n_splits=10, shuffle=True)
acc = cross_val_score(model, rescaled_X, Y, cv=fold, scoring='accuracy')
print(acc, "\n")
result = 0
for a in acc:
    result = result + a
avg = result / len(acc)
print("평균값 :", avg)
