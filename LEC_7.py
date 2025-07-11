import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import datetime as dt


# fig, ax = plt.subplots(2, 3)
# fig, ax = plt.subplots(2, 3, sharex='col', sharey='row')

# for i in range(2):
#     for j in range(3):
#         ax[i,j].text(0.5, 0.5, str((i,j)), fontsize=16, ha='center')


# grid = plt.GridSpec(2,3)

# plt.subplot(grid[:, 0])
# plt.subplot(grid[0, 1:])
# plt.subplot(grid[1, 2])
# plt.subplot(grid[1, 1])


# mean = [0,0]
# cov = [[1,2],[2,1]]
# rng = np.random.default_rng(1)

# x, y = rng.multivariate_normal( mean= mean, cov= cov, size=3000).T

# fig = plt.figure()
# grid = plt.GridSpec(4, 4, hspace= 0.2, wspace= 0.2)

# main_ax = fig.add_subplot(grid[:-1,1:])

# main_ax.plot(x, y, 'ok', markersize=2, alpha= 0.2)

# y_hist = fig.add_subplot(grid[:-1, 0], xticklabels= [], sharey = main_ax)

# x_hist = fig.add_subplot(grid[-1, 1:], yticklabels= [], sharex = main_ax)

# y_hist.hist(y, 40, orientation='horizontal', color='gray')

# x_hist.hist(x, 40, color='gray')

## Поясняющие надписи

# births = pd.read_csv('births-1969.csv')

# # births['day'] = births['day'].astype(int)

# births.index = pd.to_datetime(10000*births.year + 100*births.month + births.day, format='%Y%m%d') 

# # print(births.head())

# births_by_date = births.pivot_table('births', [births.index.month, births.index.day])

# births_by_date.index = [dt.datetime(1969, month, day) for (month, day) in births_by_date.index]

# print(births_by_date.head())

# fig, ax = plt.subplots()
# births_by_date.plot(ax=ax)

# style = dict(size=10, color='gray')
# ax.text('1969-01-01', 5500, 'Новый год', **style)
# ax.text('1969-09-01', 4500, 'День знаний', ha='right')

# ax.set(title='Рождаемость в 1969 году', ylabel='Число рождений рождения')

# ax.xaxis.set_major_formatter(plt.NullFormatter())
# ax.xaxis.set_minor_formatter(mpl.dates.DateFormatter('%h'))
# ax.xaxis.set_minor_locator(mpl.dates.MonthLocator(bymonthday=15))

# fig = plt.figure()
# ax1 = plt.axes()
# ax2 = plt.axes([0.4, 0.3, 0.1, 0.2])

# ax1.set_xlim(0,2)

# ax1.text(0.6, 0.8, 'Data1 (0.6, 0.8)', transform=ax1.transData)
# ax1.text(0.6, 0.8, 'Data2 (0.6, 0.8)', transform=ax2.transData)

# ax1.text(0.5, 0.1, 'Data3 (0.5, 0.1)', transform=ax1.transAxes)
# ax1.text(0.5, 0.1, 'Data4 (0.5, 0.1)', transform=ax2.transAxes)

# ax1.text(0.2, 0.1, 'Data5 (0.2, 0.1)', transform=fig.transFigure)
# ax1.text(0.2, 0.3, 'Data6 (0.2, 0.3)', transform=fig.transFigure)


fig, ax = plt.subplots()

x = np.linspace(0,20, 1000)
ax.plot(x, np.cos(x))
ax.axis('equal')

ax.annotate('Локальный максимум', xy = (2*np.pi, 1), xytext=(6.5,3), arrowprops=dict(facecolor='red'))

ax.annotate('Локальный минимум', xy = (np.pi, -1), xytext=(3,-3), arrowprops=dict(facecolor='blue', arrowstyle='->'))

fig, ax = plt.subplots(2,2, sharex=True, sharey=True)

for axi in ax.flat:
    axi.xaxis.set_major_locator(plt.MaxNLocator(10))
    axi.yaxis.set_major_locator(plt.MaxNLocator(3))

x = np.random.randn(1000)

plt.show()


# fig = plt.figure(facecolor=(0.9, 0.9, 0.9))
# ax = plt.axes(facecolor=(0.9, 0.85, 0.8))
# plt.grid(color='w', linestyle='solid')

# ax.xaxis.tick_bottom()
# ax.yaxis.tick_right()


# with plt.style.context('default') as something:
#     plt.hist(x, color = (0.6, 0.55, 0.5))

# ## .matplotlibrc


# plt.show()
s