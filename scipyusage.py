import scipy as sp

import matplotlib.pyplot as plt

data = sp.genfromtxt('web_traffic.tsv', delimiter='\t')

print(data[:10])

x = data[:, 0]
y = data[:, 1]

print(sp.sum(sp.isnan(y)))

x = x[~sp.isnan(y)]
y = y[~sp.isnan(y)]

print(sp.sum(sp.isnan(y)))

plt.scatter(x, y, s=5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits")  
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')

fp1, residuals, rank, sv, recond = sp.polyfit(x, y, 1, full=True)
print(fp1)
print(residuals)

# This will produce an underfitting line
# first degree polynomial
# y = ax+c
f1 = sp.poly1d(fp1)
print(f1)

fx = sp.linspace(0, x[-1], 1000)
plt.plot(fx, f1(fx), linewidth=3, color='green')
plt.legend(["d=%i" % f1.order], loc="upper left")

f2p = sp.polyfit(x,y,2)
print(f2p)

# second degree/quadratic polynomial
# y = ax**2 + bx + c
f2 = sp.poly1d(f2p)
print(f2)

plt.plot(fx, f2(fx), linewidth=3, color='black')

# This will produce an overfitting line
f53p = sp.polyfit(x,y,53)
print(f53p)

# second degree/quadratic polynomial
# y = ax**2 + bx + c
f53 = sp.poly1d(f53p)
print(f53)

plt.plot(fx, f53(fx), linewidth=3, color='red')
plt.show()

#%%

inflection = 650
xa = x[:inflection]
ya = y[:inflection]

xb = x[inflection:]
yb = y[inflection:]

fa = sp.poly1d(sp.polyfit(xa, ya, 1))
fb = sp.poly1d(sp.polyfit(xb, yb, 1))

plt.scatter(x, y, s=5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits")  
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')
plt.plot(fx, fa(fx), linewidth=3, color='red')
plt.show()

# So we cam across two terms 1) Overfitting, 2) Underfitting

#%%testing only for points abpve the 650 days

plt.scatter(x, y, s=5)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.ylabel("Hits")  
plt.autoscale(tight=True)
plt.grid(True, linestyle='-', color='0.75')


plt.plot(xb, fb(xb), linewidth=3, color='green')

fb10 = sp.poly1d(sp.polyfit(xb, yb, 10))
plt.plot(xb, fb10(xb), linewidth=3, color='black')

fb53 = sp.poly1d(sp.polyfit(xb, yb, 53))
plt.plot(xb, fb53(xb), linewidth=3, color='red')
plt.show()
