import graphviz
from uuid import uuid4


class Value():
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
                result = sum(traverse(p) for p in node._parents)
            elif node._op == '*':
                result = 1.0
                for p in node._parents:
                    result *= traverse(p)
            else:
                result = node.data
            memo[id(node)] = result
            return result
        return traverse(self)
        
    def __add__(self, other: Value):
        output = Value(self.data + other.data, _parents=(self, other), _op='+')

        def backward():
            self.grad += 1.0 * output.grad
            other.grad += 1.0 * output.grad
        self._backward = backward
        
        return output

    def __mul__(self, other: Value):
        output = Value(self.data * other.data, _parents=(self, other), _op='*')

        def backward():
            self.grad += other.data * output.grad
            other.grad += self.data * output.grad
        self._backward = backward
        
        return output

    def __repr__(self):
        return f"Value({self.label} | data={self.data} | grad={self.grad})"


def trace(root: Value):
    dot = graphviz.Digraph(comment='Backpropagation graph')
    dot.attr(rankdir='LR')

    node2id = {}
    visited: set[Value] = set()
    def traverse(node: Value | None):
        if node is None or node in visited:
            return
        nid = "%s | %s" % (uuid4(), node.data)
        nop = "%s | %s" % (uuid4(), node._op)
        dot.node(nid, label="%s | data=%.4f | grad=%.4f" % (node.label, node.data, node.grad))
        if node._op:
            dot.node(nop, label=node._op)
        node2id[node] = (nid, nop)
        visited.add(node)
        for p in node._parents:
            traverse(p)
    traverse(root)

    def connect(node: Value | None):
        if node is None:
            return
        nid, nop = node2id[node]
        if node._op:
            dot.edge(nop, nid)
        for p in node._parents:
            pid, pop = node2id[p]
            dot.edge(pid, nop)
            connect(p)
    connect(root)

    dot.render('images/graph', format='png', cleanup=True)

    return dot


if __name__ == "__main__":
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b; c.label = 'c'
    d = Value(-3, label='d')
    e = c + d; e.label = 'e'
    e.backward()
    print(a)
    print(e)
    trace(e)
