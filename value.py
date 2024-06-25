from __future__ import annotations
from typing import Optional, Tuple
import math

class Value():
    '''
    Base class for values, which can be used in computational graph and calculate gradients.
    '''
    def __init__(self, data, _children:Tuple[Value]=(), _op:str="", label:str=""):
        self.data = data
        self._op:str = _op
        self._prev:Tuple[Value] = set(_children)
        self._backward = lambda : None
        self.grad:float = 0.0
        self.label:str = label

    def __repr__(self) -> str:
        return f"Value({self.data}, label={self.label})"

    def __add__(self, other:Value) -> Value:
        #assert isinstance(other,Value)
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other) -> Value:
        return self * other

    def __neg__(self) -> Value:
        return self * -1

    def __sub__(self, other:Value) -> Value:
        return self + (-other)

    def __mul__(self, other:Value) -> Value:
        #assert isinstance(other,Value)
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad +=  other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __rmul__(self, other) -> Value:
        return self * other

    def __truediv__(self, other:Value) -> Value:
        #assert isinstance(other,Value)
        return self * other**-1
        #return Value(self.data / other.data, (self, other), "/")

    def __pow__(self, other) ->Value:
        #assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self, ),f"**{other}")

        def _backward():
            self.grad += other * self.data**(other-1) * out.grad
        out._backward = _backward
        return out

        return out
    def exp(self):
        x = self.data
        out = Value(math.exp(x),ī)
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out


    def tanh(self) -> Value:
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t,(self,),"tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out



    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()





def main():
    a = Value(1.0)
    b = Value(-3.0)
    print(a*b)



if __name__ == "__main__":
    main()

