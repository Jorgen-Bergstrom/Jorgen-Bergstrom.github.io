---
date: '2026-09-24T00:00:00-05:00'
draft: false
title: 'A Tiny Autograd Engine, Explained'
author: 'Jorgen Bergstrom'
tags: ['ML', 'Python']
---

## Introduction

`scalar_autograd.py` is a complete reverse-mode automatic differentiation engine in about 80 lines [(heres the repo)](https://github.com/Jorgen-Bergstrom/Scalar_AutoGrad). It has a single class, `Value`, that wraps one scalar and remembers how that scalar was computed, so that calling `.backward()` fills in the derivative of the final value with respect to every value that fed into it.

It's the same core idea as PyTorch's autograd, just small enough to read in one sitting. The code is also strongly influenced by Karpathy's micrograd [repo](https://github.com/karpathy/micrograd).


## What is the purpose of an Autograd Engine
Consider a case when we have three x-parameters: $x_1$, $x_2$, $x_3$.
In this example, these x-parameters have the following known values: $x_i = [2, 3, 4]$.

Let's study a problem with these equations:
$$
\begin{aligned}
f_1(x_i) &= x_1 + x_2 \\
f_2(x_i) &= x_3 \cdot f_1(x_i) \\
\mathcal{L}(x_i) &= \sin(f_2(x_i))
\end{aligned}
$$
where $\mathcal{L}$ is the loss function.
If we plug in the known values for $x_i$ then we get
$$
\begin{aligned}
f_1 &= 5\\
f_2 &= 20\\
\mathcal{L} &= \sin(20) \approx 0.91294
\end{aligned}
$$
These calculation are obviously super quick and easy to perform,
but what if we want to calculate the gradient of the loss function with respect to the variables $x_i$?
The equations for the gradient terms [$\partial{\mathcal{L}}/\partial{x_1}$,
$\partial{\mathcal{L}}/\partial{x_1}$,
$\partial{\mathcal{L}}/\partial{x_1}$] can be derived in closed-form
and then quickly calculated.
But what if the loss function $\mathcal{L}(x_i)$ depends on a million (or more) variables $x_i$.
In that case it would be **convenient** if we could automatically obtain the gradient
terms at the same time as the get the loss function value. After all, we need the gradient terms in
order to perform gradient descent to minimize the loss.

It turns out there is a clever way to calculate the gradients at the same time as the
loss function is calculated. This approach is discussed in this article,
and the approach to solve it is to create a custom Python class that keeps track
of not only numerical values, but also the local gradients for each calculation step.


## Main Python class

Every node stores the *local* derivatives of its output with respect to each of its inputs — and it computes them **during the forward pass**, while the numbers are right there.

```python
class Value:
    def __init__(self, data, _children=(), _op='', _local_grads=()):
        self.data = data
        self.grad = 0
        self._prev = list(_children)      # inputs that produced this node
        self._op = _op                    # kept for repr/debugging only
        self._local_grads = _local_grads  # ∂out/∂child_i per child
```

`_prev` holds the children (the inputs), and `_local_grads` holds one number per child: the partial derivative of *this* node's output with respect to that child. The two are positionally aligned.

Because the local derivatives are stored as plain data, the backward pass never has to know what operation created a node. It just multiplies and accumulates.

## Building the graph in the forward pass

Each operator computes its output *and* its local derivatives at the same time. For addition and multiplication that's trivial:

```python
def __add__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data + other.data, (self, other), '+', (1.0, 1.0))

def __mul__(self, other):
    other = other if isinstance(other, Value) else Value(other)
    return Value(self.data * other.data, (self, other), '*', (other.data, self.data))
```

For `a + b`, both partials are `1`. For `a * b`, the partial with respect to `a` is `b.data`, and with respect to `b` is `a.data`. Powers, ReLU, and sine follow the same pattern:

```python
def __pow__(self, other):
    return Value(self.data**other, (self,), f'**{other}',
                 (other * self.data**(other - 1),))

def relu(self):
    return Value(0 if self.data < 0 else self.data, (self,), 'ReLU',
                 (1.0 if self.data > 0 else 0.0,))

def sin(self):
    return Value(math.sin(self.data), (self,), 'sin', (math.cos(self.data),))
```

So after a forward expression like `g = f / 2.0 + 10.0 / f`, the graph is built and every edge is already labelled with its local derivative.

## The backward pass is generic

Backpropagation is the chain rule applied over the graph, plus the fact that a node used more than once gets its gradients *summed* from each use. That means we need to process nodes in an order where every node's own gradient is complete before we push it to its children — i.e. parents before children, or reverse topological order.

```python
def backward(self):
    # children before parents in topological order
    topo = []
    visited = set()

    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)

    # the entire backward pass, generic over all ops
    self.grad = 1.0
    for v in reversed(topo):
        for child, local in zip(v._prev, v._local_grads):
            child.grad += local * v.grad
```

That inner loop is the whole algorithm:

> for every node, send `local_derivative × node.grad` to each child, accumulating.

`self.grad = 1.0` seeds the derivative of the output with respect to itself (if the output is a loss, this is `dL/dL = 1`). The `+=` is what makes fan-out correct — if a value is used twice, its contributions add up. And leaf nodes simply have `_prev == []` and `_local_grads == ()`, so the `zip` is empty and they're skipped with no special case.


## Adding a new operation

Because of that, extending the engine is a one-liner. Want `exp`? Supply its local derivative and you're done — backward already handles it:

```python
def exp(self):
    return Value(math.exp(self.data), (self,), 'exp', (math.exp(self.data),))
```

## A worked check

The classic test, matching the original micrograd README:

```python
from scalar_autograd import Value

x1 = Value(2.0, (), 'x1')
x2 = Value(3.0, (), 'x2')
x3 = Value(4.0, (), 'x3')

f1 = x1 + x2
f2 = x3 * f1
Loss = f2.sin()

Loss.backward()

print(f"x1 = {x1}")
print(f"x2 = {x2}")
print(f"x3 = {x3}")
print(f"f1 = {f1}")
print(f"f2 = {f2}")
print(f"Loss = {Loss}")
```

The output from the code is:
```
x1 = Value(data=2.0, grad=1.6323282472535678, op=x1)
x2 = Value(data=3.0, grad=1.6323282472535678, op=x2)
x3 = Value(data=4.0, grad=2.0404103090669596, op=x3)
f1 = Value(data=5.0, grad=1.6323282472535678, op=+)
f2 = Value(data=20.0, grad=0.40808206181339196, op=*)
Loss = Value(data=0.9129452507276277, grad=1.0, op=sin)
```

## Two things to keep in mind

- **Gradients accumulate.** `backward()` seeds the output's grad and adds into everything else, so calling it twice without resetting will double-count. Zero the gradients between steps — that's exactly what `zero_grad()` does in the neural-net code built on top of this.
- **The graph is only as correct as the local derivatives.** If `_local_grads` and `_prev` ever get out of sync in length, `zip` silently stops at the shorter one. Keeping them constructed together, as every op here does, avoids that entirely.

That's the whole engine: label every edge with its local slope while going forward, then walk the graph backward multiplying and accumulating. Everything else is just efficiency.
