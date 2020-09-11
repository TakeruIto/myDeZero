import sys
sys.path.append('..')
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = x.reshape(6)
y.backward(create_graph=True)
print(x.grad)

y = x.T
x.cleargrad()
y.backward()
print(x.grad)
