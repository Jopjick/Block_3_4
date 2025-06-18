import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


## Трехмерные точки и линии

# fig = plt.figure()

# ax = plt.axes(projection='3d')

# z1 = np.linspace(0, 15, 10000)
# y1 = np.cos(z1)
# x1 = np.sin(z1)

# z2 = 15 * np.random.random(100)
# y2 = np.cos(z2) + 0.1 * np.random.random(100)
# x2 = np.sin(z2) + 0.1 * np.random.random(100)

# ax.plot3D(x1, y1, z1)
# ax.scatter3D(x2, y2, z2, c=z2, cmap='Greens')


# x = np.linspace(-6,6, 30)
# y = np.linspace(-6,6, 30)
# X,Y = np.meshgrid(x,y)

# def f(x,y):
#     return np.sin(np.sqrt(x**2 + y**2))

# Z = f(X,Y)

# ax.contour3D(X,Y,Z, 100)

# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')

# ax.view_init(35, -60)

# ax.scatter3D(X,Y,Z, c=Z, cmap='Greens')

## Каркасный

# ax.plot_wireframe(X,Y,Z)

## Поверхностный

# ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')

# ax.set_title('Пример')


# fig = plt.figure()

# ax = plt.axes(projection='3d')

# def f(x,y):
#     return np.sin(np.sqrt(x**2 + y**2))

# # r = np.linspace(0,6, 40)
# # theta = np.linspace(-0.9 * np.pi, 0.8 * np.pi, 40)

# # R, Theta = np.meshgrid(r, theta)
# # X = R * np.cos(Theta)
# # Y = R * np.sin(Theta)

# # Z = f(X,Y)

# # ax.plot_surface(X,Y,Z, cmap='viridis', edgecolor='none')


# ## Триангуляция поверхностей

# theta = 2 * np.pi + np.random.random(1000)
# r = 6 * np.random.random(1000)

# x = r * np.cos(theta)
# y = r * np.sin(theta)
# z = f(x,y)

# ax.scatter(x, y, z, c=z, cmap='viridis')

# ax.plot_trisurf(x, y, z)

# plt.show()




## Seaborn 
# - Dataframe (MartPlotLib с Pandas)
# - Более высокоуровневая библиотека

# data = np.random.multivariate_normal([0,0], [[5, 2],[2, 2]], size= 2000)
# data = pd.DataFrame(data, columns = ['x', 'y'])

# print(data.head())

# fig1, ax1 = plt.subplots()

# plt.hist(data['x'], alpha=0.5)
# plt.hist(data['y'], alpha=0.5)


# fig1, ax1 = plt.subplots()

# sns.kdeplot(data=data, shade=True)

# iris = sns.load_dataset('iris')
# print(iris.head())

# sns.pairplot(iris, hue='species')


# tips = sns.load_dataset('tips')
# print(tips.head())

## Гистограммы
# grid = sns.FacetGrid(tips, row='sex', col='day', hue='time')
# grid.map(plt.hist, 'total_bill', bins = np.linspace(0,40,20), alpha=0.5)
# plt.legend()

# sns.catplot(x='day', y='total_bill', data=tips, kind='box')

# sns.jointplot(x='tip', y='total_bill', data=tips, kind='hex')


# planets = sns.load_dataset('planets')
# print(planets.head())

# sns.catplot(data= planets, x= 'year', kind= 'count', order=range(2005,2015))
# sns.catplot(data= planets, x= 'year', hue= 'method', kind= 'count', order=range(2005,2015))


tips = sns.load_dataset('tips')
print(tips.head())

## Сравнение числовых данных

## Числовые пары

# sns.pairplot(tips)

## Тепловая карта

tips_corr = tips[['total_bill', 'tip', 'size']]

# sns.heatmap(tips_corr.corr(), cmap='RdBu_r', annot=True, vmin=-1, vmax=1)
## 0 - независимость; 1 - прямая кореляция; 2 - антикорреляция;

## Диаграмма рессеяния

# sns.scatterplot(data = tips, x= 'total_bill', y= 'tip', hue='sex')
# sns.relplot(data = tips, x= 'total_bill', y= 'tip', hue='sex')

## Диаграмма рессеяния + Линейная регрессия

# sns.regplot(data = tips, x= 'total_bill', y= 'tip')


## Линейный график

# sns.lineplot(data=tips, x='total_bill', y='tip')

## Сводная диаграмма

# sns.jointplot(data=tips, x='total_bill', y='tip')



## Сравнение числовых и категориальных данных

## Гистограмма
# sns.barplot(data=tips, x='total_bill', y='day', hue='sex')

# sns.pointplot(data=tips, y='total_bill', x='day', hue='sex')

## Ящик "с усами"
# sns.boxplot(data=tips, x='total_bill', y='day', hue='sex')

## Скрипичная диагармма
# sns.violinplot(data=tips, x='total_bill', y='day', hue='sex')


## Одномерная диаграмма рессеяния

sns.stripplot(data=tips, y='total_bill', x='day')

plt.show()