# matplotlib 설치
import matplotlib.pyplot as plt

#pandas 설치
import pandas as pd

# numpy 설치
import numpy as np

# scikit-learn 설치
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Binarizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 각 항목 이름 붙이기
colums = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
data = pd.read_csv('./data/pima-indians-diabetes.data.csv', names=colums)

# 바꾸기 할 항목 범위 정하기
array = data.values
X = array[:, 0:8]
Y = array[:, 8]

# 소수점 아래로 바꾸기 1 : 미니맥스스케일러
scaler = MinMaxScaler(feature_range=(0, 1))
rescaled_X = scaler.fit_transform(X)

# 데이터 분할
(X_train, X_test, Y_train, Y_test) = train_test_split(rescaled_X, Y, test_size=0.3)

# 출력
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


