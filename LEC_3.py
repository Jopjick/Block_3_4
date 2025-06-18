import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

## Pandas - расширение NumPy (структурированные массивы). Строки и столбцы индексируются метками, а не тольк очисловыми значениями
## Series, DataFrame, Index

## Series

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data)
# print(type(data))

# print(data.values)
# print(data.index)

# print(type(data.values))
# print(type(data.index))

# data = pd.Series([0.25, 0.5, 0.75, 1.0])
# print(data[0])
# print(data[1:3])

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])
# print(data)
# print(data['a'])
# print(data['b':'d'])

# print(data.index)
# print(type(data.index))

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 10, 7, 'd'])

# print(data[1])
# print(data[10: 'd'])

# population_dict = {
#     'city_1': 1000,
#     'city_2': 1001,
#     'city_3': 1002,
#     'city_4': 1003,
#     'city_5': 1004,
#     'city_6': 1005,
# }

# population = pd.Series(population_dict)

# print(population['city_4':'city_5'])

## Для создания Series можно использовать 
# 1) Списки Python и массивы из Numpy
# 2) Скалярные значения
# 3) Словари

## Q1: Привести различные способы задания объектов типа  Series

## DataFrame -- двумерный массив с явно определенными индексами. (Последовательность согласованныx по индексам объектов Series)

# area_dict = {
#     'city_1': 10000,
#     'city_2': 10010,
#     'city_3': 10020,
#     'city_4': 10030,
#     'city_5': 10040,
#     'city_6': 10050,
# }

# population_dict = {
#     'city_1': 1000,
#     'city_2': 1001,
#     'city_3': 1002,
#     'city_4': 1003,
#     'city_5': 1004,
#     'city_6': 1005,
# }

# population = pd.Series(population_dict)
# area = pd.Series(area_dict)

# states = pd.DataFrame({
#     'population': population,
#     'area': area
# })

# print(states)
# print(states.values)
# print(states.index)
# print(states.columns)

# print(type(states))
# print(type(states.values))
# print(type(states.index))
# print(type(states.columns))

# print(states['area'])

## Способы создания DataFrame 
# 1) Через объекты Series
# 2) Списки словарей
# 3) Словари объектов Series
# 4) Двумерный массив NumPy
# 5) СТруктурированный массив NumPy

## Q2: Привести различные способы создания объектов типа DataFrame

## Index -- способ организации ссылки на данные объектов Series и DataFrame. 
# Index неизменяем, упорядочен, является мультимножеством (могут быть повторяющиеся значения)

# ind = pd.Index([2,3,5,7,11])
# print(ind[1])
# print(ind[::2])

# try:
#     ind[1] = 5
# except Exception as e:
#     print(e)

## Index следуюет соглашениям объекта типа set (Python)

# indA = pd.Index([1,2,3,4,5])
# indB = pd.Index([2,3,4,5,6])
# print(indA.intersection(indB))

## Выборка данных из Series

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])

# print('a' in data)
# print('z' in data)

# print(data.keys())

# print(list(data.items()))

# data['a'] = 100
# data['z'] = 1000

# print(data)

## Объект типа Series как одномерный массив

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = ['a', 'b', 'c', 'd'])

# print(data['a':'c'])
# print(data[0:2])
# print(data[(data > 0.5) & (data < 1.0)])
# print(data[['a', 'd']])

## Аттрибуты-индексаторы

# data = pd.Series([0.25, 0.5, 0.75, 1.0], index = [1, 3, 10, 15])

# print(data[1])
# 0
# print(data.loc[1]) # значение ключа
# print(data.iloc[1]) # порядковый номер ключа

## Выборка данных из DataFrame

## DataFrame Как словарь

# area_dict = {
#     'city_1': 10000,
#     'city_2': 10010,
#     'city_3': 10020,
#     'city_4': 10030,
#     'city_5': 10040,
#     'city_6': 10050,
# }

# population_dict = {
#     'city_1': 1000,
#     'city_2': 1001,
#     'city_3': 1002,
#     'city_4': 1003,
#     'city_5': 1004,
#     'city_6': 1005,
# }

# population = pd.Series(population_dict)
# area = pd.Series(area_dict)

# data = pd.DataFrame({
#     'popl': population,
#     'pop': population,
#     'area': area,
# })

# print(data['area'])
# print(data.area)

# print(data.popl is data['popl'])

# data['new'] = data['area']

# print(data)

## DataFrame как двумерный массив

# data = pd.DataFrame({
#     'popl': population,
#     'area': area,
# })

# print(data)
# print(data.values)
# print(data.T)

# print(data['area'])
# print(data.values[0:3])

## Аттрибуты-индексаторы

# data = pd.DataFrame({
#     'popl': population,
#     'pop': population,
#     'area': area,
# })

# print(data)
# print(data.iloc[:3, 1:2])
# print(data.loc[:'city_4', 'popl':'pop'])
# print(data.loc[data['pop'] > 1002, ['area','pop']])

# data.iloc[0,2] = 9999
# print(data)

## Универсальные функции

# rng = np.random.default_rng()
# s = pd.Series(rng.integers(0,10,4))

# print(s)
# print(np.exp(s))

# area = pd.Series({
#     'city_1': 10000,
#     'city_2': 10010,
#     'city_13': 10020,
#     'city_4': 10030,
#     'city_15': 10040,
#     'city_6': 10050,
# })

# population = pd.Series({
#     'city_1': 1000,
#     'city_12': 1001,
#     'city_3': 1002,
#     'city_4': 1003,
#     'city_5': 1004,
#     'city_16': 1005,
# })

# data = pd.DataFrame({
#     'popl': population,
#     'area': area,
# })

# print(data)

## Q3: Объедените два объекта Series с неодинаковыми множествами ключей так, чтобы вместо Nan было установлено значение 1

## Объединение DataFrame-ов

# rng = np.random.default_rng()

# dfA = pd.DataFrame(rng.integers(0,10, (2,2)), columns = ['a', 'b'])
# dfB = pd.DataFrame(rng.integers(0,10, (3,3)), columns = ['a', 'b', 'c'])

# print(dfA)
# print()
# print(dfB)
# print()
# print(dfA + dfB)

rng = np.random.default_rng(1)

A = rng.integers(0, 10, (3,4))

# print(A)
# print(A[0])

# print(A - A[0])

df = pd.DataFrame(A, columns = ['a', 'b', 'c', 'd'])

# print(df)

# print(df.iloc[0])

# print(df - df.iloc[0])

# print(df.iloc[0, ::2])

# print(df - df.iloc[0, ::2])

## Q3: Переписать пример с транслированием для DataFrame так, чтобы вычитание происходило не по строкам, а по столбцам

# NA-значения: NaN, null, -99999999

# sum / nansum обрабатывает Nan

# Pandas. Два способа хранения отсутсвующих значений
# 1) Индикаторы Nan, none
# 2) null

# None - объект, его использование может привести к накладным расходам
# Не работает с агрегирующими операторами

# val1 = np.array([1,2,3])
# print(val1.sum())

# try:
#     val1 = np.array([1,None,3])
#     print(val1.sum())
# except:
#     pass

# val1 = np.array([1,np.nan,3])
# print(val1)
# print(val1.sum())
# print(np.sum(val1))
# print(np.nansum(val1))

# NaN - числа

# x = pd.Series(range(10), dtype=int)
# print(x)

# x[0] = None
# x[1] = np.nan

# print(x)

# x1 = pd.Series(['a', 'b', 'c'])
# print(x1)

# x1[0] = None
# x1[1] = np.nan

# print(x1)

## 

# x2 = pd.Series([1,2,3, np.nan, None, pd.NA])
# print(x2)

# x3 = pd.Series([1,2,3, np.nan, None, pd.NA], dtype='Int32')
# print(x3)

# print(x3.isnull())
# print(x3[x3.isnull()])
# print(x3[x3.notnull()])

# print(x3.dropna())

# df = pd.DataFrame([
#     [1,2,3, np.nan, None, pd.NA],
#     [1,2,3,4,5,6],
#     [1, np.nan,3,4, np.nan,6]
# ])

# print(df)
# print()
# print(df.dropna())
# print()
# print(df.dropna(axis=0))
# print()
# print(df.dropna(axis=1))

# how
# all - все значения NA
# any - хотя бы одно значение NA

# thresh=x -  остается, если присутсвует минимум x непустых значений

# print(df.dropna(axis=1, how='all'))

# print(df.dropna(axis=1, how='any'))

# print(df.dropna(axis=1, thresh=2))

## Q4: На примере DataFrame продемонстрируйте использование методов 