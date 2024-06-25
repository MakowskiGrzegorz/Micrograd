from value import Value
import numpy as np
from typing import Union
class Neuron:
    def __init__(self, num_inputs:int):
        self.w:list[Value] = [Value(np.random.uniform(-1,1)) for _ in range(num_inputs)]
        self.b:Value = Value(np.random.uniform(-1,1))

    def __call__(self, x:list[Union[int,float]])-> list[Union[int,float]]:
#        assert len(x) == len(self.w)
        #out = sum([self.w[i]*x[i] for i in range(len(self.w))]) + self.b
        act = sum((wx * wi for wx,wi in zip(self.w,x)), self.b)
        out  = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]
class Layer:
    def __init__(self,num_inputs:int, num_outputs:int):
        self.neurons:list[Neuron] = [Neuron(num_inputs) for _ in range(num_outputs)]
        self.num_inputs:int = num_inputs

    def __call__(self, x:list[Union[int,float]])->Union[list[Union[int,float]],Union[int,float]]:
#        assert len(x) == self.num_inputs
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]
class MLP:
    def __init__(self, num_inputs:int, num_outputs:int, hidden_dims:list[int]=[]):
        '''
        Implementation of multilayer perceptron, which consists of multiple stacked layers.
        '''
        x = [num_inputs] + hidden_dims + [num_outputs]
        self.layers = [Layer(x[i],x[i+1]) for i in range(len(x)-1)]

    def __call__(self, x:list[Union[int,float]])->Union[list[Union[int,float]],Union[int,float]]:
        for layer in self.layers:
            x = layer(x)
        return x
    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
def l2_loss(y,ypred):
    #return [(yi,ypred) for yi, ypred in zip(y,ypred)]
        return sum((ypred - yi)**2 for yi, ypred in zip(y,ypred))
if __name__ == '__main__':
    xs = [
        [2.0,3.0,-1.0],
        [3.0,-1.0,0.5],
        [0.5,1.0,1.0],
        [1.0,1.0,-1.0]
        ]
    y = [1.0,-1.0,-1.0-1.0]

    nn = MLP(len(xs[0]),1,[4,4])

    xs = [
    [2.0,3.0,-1.0],
    [3.0,-1.0,0.5],
    [0.5,1.0,1.0],
    [1.0,1.0,-1.0]
    ]
    y = [1.0,-1.0,-1.0,-1.0]

    ypred = [nn(x) for x in xs]

    loss = sum((ypred - yi)**2 for yi, ypred in zip(y,ypred))
    loss.backward()

    for i in range(10):
        # forward pass
        y_pred = [nn(x) for x in xs]
        loss = l2_loss(y, y_pred)

        # backward pass
        for p in nn.parameters():
            p.grad = 0.0
        loss.backward()

        # update
        for p in nn.parameters():
            p.data += -0.01 * p.grad
