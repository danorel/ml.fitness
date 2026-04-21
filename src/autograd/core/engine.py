import math


class Value:
    def __init__(self, data: float, _parents=(), _op='', label=''):
        self.data: float = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._parents: tuple[Value] = _parents
        self._op = _op

    def backward(self):
        visited = set()
        nodes = []

        def traverse(node: Value | None):
            if node is None or node in visited:
                return
            visited.add(node)
            for p in node._parents:
                traverse(p)
            nodes.append(node)
        traverse(self)

        self.grad = 1.0
        for node in reversed(nodes):
            node._backward()

    def compute(self, override_node=None, override_delta=0.0):
        memo = {}
        def traverse(node: Value) -> float:
            if id(node) in memo:
                return memo[id(node)]
            if node is override_node:
                result = node.data + override_delta
            elif not node._parents:
                result = node.data
            elif node._op == '+':
                result = traverse(node._parents[0]) + traverse(node._parents[1])
            elif node._op == '-':
                result = traverse(node._parents[0]) - traverse(node._parents[1])
            elif node._op == '*':
                result = traverse(node._parents[0]) * traverse(node._parents[1])
            elif node._op == '/':
                result = traverse(node._parents[0]) / traverse(node._parents[1])
            elif node._op == '**':
                result = traverse(node._parents[0]) ** traverse(node._parents[1])
            elif node._op == 'tanh':
                result = math.tanh(traverse(node._parents[0]))
            elif node._op == '-1':
                result = -traverse(node._parents[0])
            else:
                result = node.data
            memo[id(node)] = result
            return result
        return traverse(self)

    def tanh(self):
        output = Value(math.tanh(self.data), _parents=(self,), _op='tanh')

        def backward():
            """
             f(w) = tanh(w) = (exp(w) - exp(-w)) / (exp(w) + exp(-w))
            f'(w) = 1 - tanh(w)**2
            """
            self.grad += (1 - output.data ** 2) * output.grad
        output._backward = backward

        return output

    def _cast(self, x):
        return x if isinstance(x, Value) else Value(x)

    def __add__(self, other: Value):
        other = self._cast(other)

        output = Value(self.data + other.data, _parents=(self, other), _op='+')

        def backward():
            """
             f(w) = w + b
            f'(w) = 1.0
            f'(b) = 1.0
            """
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = backward

        return output

    def __radd__(self, other):
        return self + other

    def __sub__(self, other: Value):
        other = self._cast(other)

        output = Value(self.data - other.data, _parents=(self, other), _op="-")

        def backward():
            """
             f(w) = w - b
            f'(w) = 1.0
            f'(b) = -1.0
            """
            self.grad += 1.0 * output.grad
            other.grad += -1.0 * output.grad
        output._backward = backward

        return output

    def __rsub__(self, other):
        other = self._cast(other)

        output = Value(other.data - self.data, _parents=(other, self), _op='-')

        def backward():
            """
             f(w) = b - w
            f'(w) = -1.0
            f'(b) = 1.0
            """
            self.grad += -1.0 * output.grad
            other.grad += 1.0 * output.grad
        output._backward = backward

        return output

    def __mul__(self, other: Value):
        other = self._cast(other)

        output = Value(self.data * other.data, _parents=(self, other), _op='*')

        def backward():
            """
             f(w) = w * b
            f'(w) = b
            f'(b) = w
            """
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        output._backward = backward

        return output

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: Value):
        other = self._cast(other)

        output = Value(self.data / other.data, _parents=(self, other), _op="/")

        def backward():
            """
             f(w) = w / b = w * (1 / b) = w * b**(-1)
            f'(w) = (1 / b)
            f'(b) = -w * b**(-2)
            """
            self.grad += (1 / other.data) * output.grad
            other.grad += (-self.data * other.data**(-2)) * output.grad
        output._backward = backward

        return output

    def __rtruediv__(self, other):
        other = self._cast(other)

        output = Value(other.data / self.data, _parents=(self, other), _op="/")

        def backward():
            """
             f(w) = b / w = b * (1 / w) = b * w**(-1)
            f'(w) = -b * w**(-2)
            f'(b) = (1 / w)
            """
            self.grad += (-other.data * self.data**(-2)) * output.grad
            other.grad += (1 / self.data) * output.grad
        output._backward = backward

        return output

    def __pow__(self, other: Value):
        other = self._cast(other)

        output = Value(self.data ** other.data, _parents=(self, other), _op="**")

        def backward():
            """
             f(w) = w**b
            f'(w) = b * w**(b - 1)
            f'(b) = w**b * ln(w)
            """
            self.grad += (other.data * self.data ** (other.data - 1)) * output.grad
            if self.data > 0:
                other.grad += (self.data ** other.data) * math.log(self.data) * output.grad
        output._backward = backward

        return output

    def __neg__(self):
        output = Value(-self.data, _parents=(self,), _op="-1")

        def backward():
            """
             f(w) = -w
            f'(w) = -1.0
            """
            self.grad += -1.0 * output.grad
        output._backward = backward

        return output

    def __repr__(self):
        return f"Value({self.label} | data={self.data} | grad={self.grad})"
