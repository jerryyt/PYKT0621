from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot

X, y = make_regression(n_samples=1000, n_features=10, n_informative=7)
print(X.shape)
r1 = LinearRegression()
r1.fit(X, y)
print(r1.coef_)
importance = r1.coef_
# iterate coef_
for i, v in enumerate(importance):
    print(f"feature{i:d}, score={v:.2f}")
pyplot.bar([x for x in range(len(importance))], importance)
pyplot.show()