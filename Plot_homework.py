import csv
import matplotlib.pyplot as plt


x = []
y = []
z = []
a = []
b = []
c = []
d = []
e = []
f = []
g = []
h = []

with open('diabetes.data','r') as csvfile:
    plots = csv.reader(csvfile, delimiter='\t')
    x = []
    y = []
    for row in plots:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(float(row[2]))
        a.append(float(row[3]))
        b.append(float(row[4]))
        c.append(float(row[5]))
        d.append(float(row[6]))
        e.append(float(row[7]))
        f.append(float(row[8]))
        g.append(float(row[9]))
        h.append(float(row[10]))

# 2d plot == to replicate (change columns)
plt.figure(2)
plt.plot(x,y, label='Diabetes Data')
plt.xlabel('Age')
plt.ylabel('sex')
plt.title('Graph Age x Sex')
plt.legend()
plt.show()

# 3d graph example 1
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(x, y, z, c=z, cmap='viridis', linewidth=0.5)

# 3d graph example
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_trisurf(c, d, e, cmap='viridis', edgecolor='none');

