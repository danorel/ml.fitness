from core.engine import Value
from core.ui import show

if __name__ == "__main__":
    a = Value(-2.0, label='a')
    b = Value(3.0, label='b')
    c = a * b; c.label = 'c'
    d = Value(-3, label='d')
    e = c + d; e.label = 'e'
    e.backward()
    print(a)
    print(e)
    show(e)
