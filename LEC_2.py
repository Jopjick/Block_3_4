import numpy as np
import matplotlib.pyplot as plt

## Суммирование значений (и другие агрегатные функции)

# rng = np.random.default_rng(1)
# s = rng.random(50)

# print(s)
# print(sum(s))
# print(np.sum(s))

# a = np.array([[1,2,3,4,5],
#              [6,7,8,9,10]])

# print(np.sum(a))
# print(np.sum(a, axis=0))
# print(np.sum(a, axis=1))

# print()

# print(np.min(a))
# print(np.min(a, axis=0))
# print(np.min(a, axis=1))

# print()

# print(a.min())
# print(a.min(0))
# print(a.min(1))

# print()

# # NaN = Not a Number
# print(np.nanmin(a))
# print(np.nanmin(a, axis=0))
# print(np.nanmin(a, axis=1))

## Транслирование (broadcasting)
## Некоторый набор правил, которы епозволяют осуществять бинарные операции с массивами разных форм и размеров

# a = np.array([0, 1, 2])
# b = np.array([5, 5, 5])

# print(a + b)
# print(a + 5) # "5" транслируется в [5,5,5], т. е. она подстраивается под размер массива a

# a = np.array([[0, 1, 2],
#               [3, 4, 5]])
# print(a + 5)

# a = np.array([0, 1, 2])
# b = np.array([[0], [1], [2]])
# print(a+b)

# Правила
# 1) Если рамерности массивов отличаются, то форма массива с меньшей размерностью дополняется 1 с левой стороны (ndim)
# 2) Если формы массивов отличаются в каком-то измерении, то если у массива форма равна 1, он "растягивается" до соответсвия формы  (shape)
# 3) Если после применения 1) и 2) в каком-либо измерении размеры отличаются и ни один из них не равен 1, тогда транслирование невозможно, генерируется ошибка

# a = np.array([[0,1,2], [3,4,5]])
# b = np.array([5])

# # print(a.ndim, a.shape)
# # print(b.ndim, b.shape)

# # a         (2,3)
# # b (1,) -> (1,1) -> (2,3)

# # print(a+b)


# a = np.ones((2,3))
# b = np.arange(3)

# print(a)
# print(b)

# print(a.ndim, a.shape)
# print(b.ndim, b.shape)

# # (2,3)   (2,3)    (2,3)
# # (3,) -> (1,3) -> (2,3)

# c = a + b

# print(c, c.shape)

# a = np.arange(3).reshape(3,1)
# b = np.arange(3)

# print(a, b, sep='\n')

# # (3,1)   (1,3) -> (3,3)
# # (3,) -> (1,3) -> (3,3)

# c = a + b

# # [0, 0, 0]   [0, 1, 2]
# # [1, 1, 1] + [0, 1, 2]
# # [2, 2, 2]   [0, 1, 2]

# print(c, c.shape)

# a = np.ones((3,2))
# b = np.arange(3)

# # 2 (3,2)    (3,2)    (3,2) -> ошибка
# # 1 (3,)  -> (1,3) -> (3,3)

# try:
#     c = a + b
# except Exception as err:
#     print(err)

## Q1: Что надо оизменить в последнем примере, чтобы он работал без ошибок?

# Центрирование массива
# X = np.array([[1,2,3,4,5,6,7,8,9],
#               [9,8,7,6,5,4,3,2,1]])

# X_mean0 = X.mean(0)
# print(X_mean0)

# X_center0 = X - X_mean0
# print(X_center0)

# X_mean1 = X.mean(1)
# print(X_mean1)

# X_mean1 = X_mean1[:, np.newaxis]

# X_center1 = X - X_mean1
# print(X_center1)

# x = np.linspace(0, 5, 50)
# y = np.linspace(0, 5, 50)[:, np.newaxis]

# z = np.sin(x)**3 + np.cos(20 + y*x) * np.sin(y)

# plt.imshow(z)
# plt.colorbar()
# plt.show()

# x = np.array([1,2,3,4,5])
# y = np.array([[1,2,3,4,5],
#             [6,7,8,9,10]])
# print(x < 3)
# print(np.less(x,3))

# # False = 0
# # True = 1

# print(np.sum(x < 3)) # количество элементов
# print(np.sum(y < 4)) # количество элементов
# print(np.sum(y < 4, axis = 0)) # количество элементов
# print(np.sum(y < 4, axis = 1)) # количество элементов

# & | ^ - -- побитовые операции

## Q2: Пример для y. Вычислить количество элементов по обоим размерностям, значения которых больше 3 и меньше 5

## Маски - булевы массивы

# x = np.array([1,2,3,4,5])
# print(x < 3)

# y = x[x < 3]
# print(y)

# print(bin(42))
# print(bin(59))
# print(bin(42 & 59))

## Векторизация индекса

# x = np.array([1,2,3,4,5,6,7,8,9])

# index = [1,5,7]
# print(x[index])

# index = [[1,5,7],
#         [2,4,8]]

# print(x[index])

## Форма результата отражает форму массива индексов, а не форму исходного массива

# x = np.arange(12).reshape((3,4))

# print(x)

# print(x[2])
# print(x[2, [2,0,1]])
# print()

# print(x[1:, [2,0,1]])

# x = np.arange(10)
# i = np.array([2,1,8,4])

# print(x)

# x[i] = 999
# print(x)

## Сортировка массивов

# x = [3,2,5,6,3,2,4,7,9,6,6,4,2,1,5,8,9]

# print(sorted(x))
# print(np.sort(x))

## Структурированные массивы

# data = np.zeros(4, dtype = {'names':('name', 'age'),
#                             'formats': ('U10', 'i4')})

# name = ['name1', 'name2', 'name3', 'name4']
# age = [10, 20, 30, 40]

# data['name'] = name
# data['age'] = age

# print(data[data['age'] > 20])

# ## Массивы записей

# data_rec = data.view(np.recarray)
# print(data_rec)
# print(data_rec[0])
# print(data_rec[-1].name)