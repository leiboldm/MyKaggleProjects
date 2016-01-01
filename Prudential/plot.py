import pandas
import sys
import matplotlib.pyplot as plt

col = sys.argv[1]

df = pandas.read_csv('train.csv')
x = df[col].get_values()

xmin = min(x)
xmax = max(x)
intx = (xmax - xmin) / 39.0

y = list()
x = list()
area = list()

for i in range(0, 40):
    ys = df[(df[col] >= xmin + i * intx) & (df[col] < xmin + (i + 1) * intx)]['Response'] 
    if len(ys) > 3:
        yavg = sum(ys.get_values()) / float(len(ys))
        y.append(yavg)
        x.append(xmin + i * intx)
        area.append(len(ys))

plt.scatter(x, y, s=area, alpha=0.5)
plt.show()
