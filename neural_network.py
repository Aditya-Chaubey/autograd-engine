from autograd_engine import Scalar

class Neuron:

    def __init__(self, nin):
        self.w = [Scalar(np.random.uniform(-1,1)) for _ in range(nin)]
        self.b = Scalar(np.random.uniform(-1, 1))

    def __call__(self, xin):
        out = sum(wi * xi for wi,xi in zip(self.w, xin))
        out += self.b
        return out
    
    def __repr__(self):
        return f"Weights : {self.w}\nBias : {self.b}"

    def parameters(self):
        return self.w + [self.b]

class Layer:

    def __init__(self, nin, nneurons):
        self.neurons = [Neuron(nin) for _ in range(nneurons)]

    def __call__(self, xin):
        return [n(xin) for n in self.neurons]
    
    def __repr__(self):
        return f"Neurons : {self.neurons}"

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:

    def __init__(self, nin, llayers):
        sz = [nin] + llayers
        self.layers = [Layer(sz[i], sz[i + 1])for i in range(len(llayers))]

    def __call__(self, xin):
        for layer in self.layers:
            xin = layer(xin)
        return xin

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
    
    def __repr__(self):
        return f"Layers: \n{self.layers}"
        