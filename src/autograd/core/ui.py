import graphviz
from uuid import uuid4

from autograd.core.engine import Value


def show(root: Value):
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