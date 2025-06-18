import matplotlib.pyplot as plt
import numpy as np

## Способы запуска кода на python
## 1) Сценарий
## 2) Командная оболочка IPython
## 3) Jupyter

## 1 Вариант
## plt.show() - запускается только 1 раз
## В процессе работы код создает объекты класса figure
## Все активыне объекты класса figure открываются в оканх

# x = np.linspace(0,10,100)
# plt.plot(x, np.sin(x))

# plt.show()

# plt.plot(x, np.cos(x))

# ## Отображается только 1 график

# plt.plot(x, np.sin(x))
# plt.plot(x, np.cos(x))

# plt.show()



## 2 Вариант

# %matplotlib
# import matplotlib.pyplot as plt
# После этого любой вывод команды plt.plot(...) будет открывать окно графика
# Очень редко plt.draw(...)



## 3 Вариант

## Похоже на  2 вариант

## %matplotlib inline - в блокнот добавляются статическая картинка
## %matplotlib notebook - в блокнот добавляются интерактивные графики



## Сохраниение картинок в файлы

# fig = plt.figure()
# fig.savefig('saved_image_1.png')

#№ Расширения
# print(fig.canvas.get_supported_filetypes())

## 2 способа выхода графиков:
## 1) MATLAB-подобный стиль 
## 2) ООП-стиль

# 1)

# plt.figure()

# plt.subplot(2, 1, 1)
# x = np.linspace(0,10, 100)
# plt.plot(x, np.sin(x))

# plt.subplot(2, 1, 2)
# plt.plot(x, np.cos(x))

# plt.show()

# 2)

## fig: Figure, ax: Axes

# x = np.linspace(0,10, 100)
# fig, ax = plt.subplots(2)

# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.cos(x))

# plt.show()

## fig: Figure - контейнер, который содержит все объекты (Система коордиант, текст, метки)
## ax: Axes - Система коориднат - прямоугольник, деления, метки. Непосредственно тут располагается график

## Цвета линий - color
## 'blue', ...
## 'rgbcmyk' -> 'rg'
## '0.14' - градация серого от 0 до 1
## RRGGBB - '#FF00EE'
## (1.0, 0.2, 0.3) - (R,G,B)

# x = np.linspace(0,10, 100)
# fig = plt.figure()
# ax = plt.axes()

# ax.plot(x, np.sin(x), color = 'blue'); # ; избавляет от текстового вывода
# ax.plot(x, np.sin(x-1), color = 'g');
# ax.plot(x, np.sin(x-2), color = '0.75');
# ax.plot(x, np.sin(x-3), color = '#FF00EE');
# ax.plot(x, np.sin(x-4), color = 'salmon');
# ax.plot(x, np.sin(x-5), color = (1.0, 0.2, 0.3));

# plt.show()

## Стиль линий - linestyle
## сплошная - '-' / 'solid'
## штрихованная - '--' / 'dashed'
## штрихпунктирная - '-.' / 'dashdot'
## пунктирная - ':' / 'dotted'

# x = np.linspace(0,5, 100)
# fig = plt.figure()
# ax = plt.axes()

# # ax.plot(x, np.sin(x), linestyle='solid')
# # ax.plot(x, np.sin(x-1), linestyle='dashed')
# # ax.plot(x, np.sin(x-2), linestyle='dashdot')
# # ax.plot(x, np.sin(x-3), linestyle='dotted')


# ax.plot(x, np.sin(x), ':g') # комбинация
# ax.plot(x, np.sin(x-1), '--k')

# plt.show()


## Операции с осями

# x = np.linspace(0,10, 100)
# fig, ax = plt.subplots(4)

# ax[0].plot(x, np.sin(x))
# ax[1].plot(x, np.sin(x))
# ax[2].plot(x, np.sin(x))
# ax[3].plot(x, np.sin(x))

# ax[1].set_xlim(-2,12)
# ax[1].set_ylim(-1.5,1.5)

# # Отзеркаливание
# ax[2].set_xlim(12,-2)
# ax[2].set_ylim(1.5,-1.5)

# # Масштабирование
# ax[3].autoscale(tight = True)

# plt.show()


# Названия и метки

# x = np.linspace(0,10, 100)

# plt.subplot(3,1,1)
# plt.plot(x, np.sin(x))

# plt.title('Синус')
# plt.xlabel('x')
# plt.ylabel('sin(x)')

# plt.subplot(3,1,2)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), '--b', label='cos(x)')

# plt.title('Синус и косинус')
# plt.xlabel('x')
# plt.legend()

# plt.subplot(3,1,3)
# plt.plot(x, np.sin(x), '-g', label='sin(x)')
# plt.plot(x, np.cos(x), '--b', label='cos(x)')

# plt.axis('equal')
# plt.title('Синус и косинус')
# plt.xlabel('x')
# plt.legend()

# plt.subplots_adjust(hspace = 0.5)

# plt.show()

# x = np.linspace(0,10, 30)
# plt.plot(x, np.sin(x), '.') # 'o' - маркер
# plt.plot(x, np.sin(x) + 1, '>')
# plt.plot(x, np.sin(x) + 2, '^')
# plt.plot(x, np.sin(x) + 3, 'o')
# plt.plot(x, np.sin(x) + 4, 's')

# plt.show()


# x = np.linspace(0,10, 30)
# # plt.plot(x, np.sin(x), '--p', markersize=15, linewidth=4,
# #          markerfacecolor='w', markeredgecolor='g', markeredgewidth=4)

# rng = np.random.default_rng(30)

# colors = rng.random(30)
# sizes = 80 * rng.random(30)

# plt.scatter(x, np.sin(x), marker='o', c = colors, s = sizes)
# plt.colorbar()
# plt.show()

## Если точек много, то plot предпочтительней из-за производительности

## Визуализация погрешности

# N = 50
# x = np.linspace(0,10,N)

# dy = 0.4
# y = np.sin(x) + dy*np.random.randn(N) 

# plt.errorbar(x, y, yerr=dy, fmt='.k')
# plt.fill_between(x, y-dy, y+dy, color = 'red', alpha=0.4)

# plt.show()

def f(x,y):
    return np.sin(x) ** 5 + np.cos(20 + x * y) * np.cos(x)

x = np.linspace(0, 5, 40)
y = np.linspace(0, 5, 40)

X, Y = np.meshgrid(x, y)

Z = f(X,Y)

# plt.contour(X, Y, Z, cmap ='RdGy')
# plt.contourf(X, Y, Z, cmap ='RdGy')
# plt.imshow(Z, extent=[0,5,0,5], cmap ='RdGy')
# plt.imshow(Z, extent=[0,5,0,5], cmap ='RdGy', interpolation='Gaussian', origin='lower')
c = plt.contour(X, Y, Z, cmap ='RdGy')
plt.clabel(c)
plt.imshow(Z, extent=[0,5,0,5], cmap ='RdGy', interpolation='Gaussian',
            origin='lower', aspect='equal')
plt.colorbar()

plt.show()