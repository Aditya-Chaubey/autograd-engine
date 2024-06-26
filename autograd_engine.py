import math
import numpy as np

class Scalar:

    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self._prev = set(_children)
        self.grad = 0.0
        self._backward = lambda : None

    def __repr__(self):
        return f"Scalar(data: {self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other
    
    def __mul__(self, other):
        other = other if isinstance(other, Scalar) else Scalar(other)
        out = Scalar(self.data * other.data, (self, other), '*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return (self * -1)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "Only supporting int and float powers"
        out = Scalar(self.data ** other)

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        d = self.data

        def _backward():
            self.grad += (d > 0) * out.grad
        
        out._backward = _backward
        out = Scalar(d * (d > 0))
        return out

    def backward(self):
        self.grad = 1.0
        self._backward()