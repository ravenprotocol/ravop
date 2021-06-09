import time
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target.reshape(-1, 1)

import ravop.core as R

from ravcom import globals as g
g.eager = True

a = R.Tensor(X)
b = R.Tensor(y)

start_time = time.time()
c = a+b
print(c.eval())
print(time.time()-start_time)

start_time = time.time()
d = a-b
print(d.eval())
print(time.time()-start_time)

start_time = time.time()
e = d*b
print(e.eval())
print(time.time()-start_time)





