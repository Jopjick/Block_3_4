# Метод опорных векторов (SCM - support vector machine) - классификация и регрессия
# Разделяющая классификация (метод опорных векторов использует ей вместо генеративной классификации)
# Выбирается линия с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC

iris = sns.load_dataset('iris')

data = iris[['sepal_length', 'petal_length', 'species']]
data_df = data[(data['species'] == 'setosa') | (data['species'] == 'versicolor')]
X = data_df[['sepal_length', 'petal_length']]
Y = data_df['species']

data_df_setosa = data_df[data['species'] == 'setosa']
data_df_versicolor = data_df[data['species'] == 'versicolor']

plt.scatter(data_df_setosa['sepal_length'], data_df_setosa['petal_length'])
plt.scatter(data_df_versicolor['sepal_length'], data_df_versicolor['petal_length'])

model = SVC(kernel='linear', C=10000)
model.fit(X, Y)


x1_p = np.linspace(min(data_df['sepal_length']), max(data_df['sepal_length']), 100)
x2_p = np.linspace(min(data_df['petal_length']), max(data_df['petal_length']), 100)
X1_p, X2_p = np.meshgrid(x1_p, x2_p)
X_p = pd.DataFrame(np.vstack([X1_p.ravel(), X2_p.ravel()]).T, columns=['sepal_length', 'petal_length'])
Y_p = model.predict(X_p)
X_p['species'] = Y_p

a = model.support_vectors_
plt.scatter(a[:,0], a[:,1], s=400, facecolor='None', edgecolor='black')

X_setosa = X_p[X_p['species'] == 'setosa']
X_versicolor = X_p[X_p['species'] == 'versicolor']

plt.scatter(X_setosa['sepal_length'], X_setosa['petal_length'], alpha=0.05)
plt.scatter(X_versicolor['sepal_length'], X_versicolor['petal_length'], alpha=0.05)

# plt.show()

# ДЗ
# Из набора IRIS убрать часть данных, на которых происходит обучение и убедиться, что на предсказание влияют только опорные вектора

frac_to_remove = 0.1  # 10% данных
indices_to_remove = np.random.choice(data_df.index, size=int(len(data_df) * frac_to_remove), replace=False)
df_dropped = data_df.drop(indices_to_remove)
dropped_df_setosa = data_df[data['species'] == 'setosa']
dropped_df_versicolor = data_df[data['species'] == 'versicolor']
plt.subplots(0,1)

# Что будет, если группы делятся плохо?