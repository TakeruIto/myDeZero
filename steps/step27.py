import math
import numpy as np
import sys
sys.path.append('..')
from dezero import Function
from dezero import Variable
from dezero.utils import plot_dot_graph

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001, loop=100000):
    y = 0
    for i in range(loop):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

if __name__ == "__main__":
    x = Variable(np.array(np.pi/4))
    y = my_sin(x, threshold=1e-150)
    y.backward()

    print(y)
    print(x.grad)

    plot_dot_graph(y, verbose=False, to_file='my_sin.png')
