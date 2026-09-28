---
date: '2026-09-26T00:00:00-05:00'
draft: false
title: 'A Tiny Neural Network, Explained'
description: 'Build a small neural-network library (Neuron, Layer, MLP) on top of a scalar autograd engine, with backpropagation for free.'
author: 'Jorgen Bergstrom'
bluesky: true
tags: ['ML', 'Neural Networks', 'Python']
---

## Introduction

In the [previous article](/posts/scalar_autograd/) we built `engine.py`: a complete reverse-mode automatic differentiation engine in about 80 lines, built around a single `Value` class. That engine knows how to record arithmetic on scalars and hand back the gradient of the final result with respect to every scalar that fed into it.

This article builds the next layer on top: `nn.py`, a small neural-network library in roughly 60 lines [(heres the repo)](https://github.com/Jorgen-Bergstrom/Scalar_AutoGrad). The punchline is that **there is no new math here**. A neuron is a weighted sum plus a nonlinearity, and a network is a stack of neurons. Because every weight is a `Value`, the forward pass silently builds an autograd graph, and one call to `.backward()` fills in the gradient of the loss with respect to every weight and bias in the network.

The whole file is three classes — `Neuron`, `Layer`, and `MLP` — plus two imports:

```python
import random
from engine import Value
```

## From one scalar to a network

Recall that a `Value` can be either a *leaf* (something we chose, like a weight) or an *intermediate* (something computed). Leaves have no children; intermediates store their children and the local derivatives computed during the forward pass.

A neural network is just a lot of leaves combined with arithmetic. If we write the network's output as a function of its inputs and weights, the engine takes care of differentiating that function for us. So all `nn.py` has to do is express the network as arithmetic on `Value`s.

## The Neuron

A single neuron computes a weighted sum of its inputs, adds a bias, and optionally applies a nonlinearity:

$$a = \phi\left(b + \sum_{i=1}^{n} w_i x_i\right)$$

where $\phi$ is the [ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)) function $\max(0, z)$, or the identity for a linear neuron. In code:

```python
class Neuron():

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
```

A few details are worth pausing on.

**The weights are `Value`s.** `self.w` holds `nin` of them and `self.b` holds the bias. These are the leaves we will eventually train, so they are created as `Value` objects from the start. The bias starts at zero; the weights are drawn from a uniform distribution on $[-1, 1]$.

**The forward pass is ordinary Python arithmetic.** `self.w` and the input `x` are zipped together and each pair is multiplied. This is where `Value.__mul__` kicks in and starts recording the graph. The built-in `sum` then adds the products, starting from `self.b`:

```python
sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
```

The second argument to `sum` is its starting value, so this is just $b + w_1 x_1 + w_2 x_2 + \dots$. It works because `Value` implements `__add__` (and `__radd__`), so the standard library never has to know it is adding autograd nodes. If you have never seen it before, the `sum(..., start)` form is a neat way to fold a sequence without a hand-written loop.

**`parameters()` returns the trainable leaves.** For one neuron that is the weight vector plus the bias. This method is the bridge to the engine: these are exactly the `Value`s whose `.grad` we will read after calling `.backward()`.

> One caveat: `zip` stops at the shorter of the two sequences, so `x` must have the same length as `self.w`. If you feed a neuron the wrong number of inputs it will silently ignore the extras rather than raise.

## The Layer

A layer is just a list of neurons that all see the same input:

```python
class Layer():

    def __init__(self, nin, nout, **kwargs):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]
```

`nin` and `nout` are the input and output widths. The `**kwargs` are forwarded to each `Neuron`, which is how the `MLP` below toggles the nonlinearity per layer.

The only slightly clever line is:

```python
return out[0] if len(out) == 1 else out
```

When a layer has a single neuron — as the output layer always does in our classifier — we unwrap the one-element list and return the `Value` itself. That lets `MLP.__call__` return a plain scalar for the regular network case, while a multi-neuron layer returns a list of `Value`s to feed the next layer. Keeping both return shapes means the final layer can behave like a function worth calling `.backward()` on.

`parameters()` uses a nested comprehension to flatten the list of neurons into one flat list of weights and biases.

The `__repr__` groups identical neurons so a wide layer prints compactly:

```python
def __repr__(self):
    counts = {}
    for n in self.neurons:
        s = repr(n)
        counts[s] = counts.get(s, 0) + 1
    parts = [f'{c} x {s}' if c > 1 else s for s, c in counts.items()]
    return f'Layer({self.nin} -> {self.nout}) [{", ".join(parts)}]'
```

So a layer of three ReLU neurons prints as `Layer(2 -> 3) [3 x ReLUNeuron(2)]` instead of listing each one.

## The MLP

The multi-layer perceptron wires the layers together:

```python
class MLP():

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def __repr__(self):
        layers = '\n'.join(f'  {i}: {layer}' for i, layer in enumerate(self.layers))
        return f'MLP of [\n{layers}\n]'
```

`nouts` is the width of each layer, from first to last. Prepending `nin` gives the full list of sizes `sz`, and each consecutive pair becomes a `Layer`.

The one design decision to notice is the nonlinearity flag:

```python
nonlin=i != len(nouts) - 1
```

Every layer except the last gets a ReLU. The last layer is **linear**, so the network outputs a raw real number — a *logit* — rather than a squashed probability. This is deliberate: the [`cancer_prediction.py`](https://github.com/Jorgen-Bergstrom/Scalar_AutoGrad) script that builds on this code trains with binary cross-entropy *with logits*, which is numerically stable when the logit saturates. We will come back to that in the next article.

`__call__` is a fold: it threads the input through each layer in turn, and each layer's output becomes the next layer's input.

`zero_grad()` exists because, as the engine article warned, **gradients accumulate**. `backward()` adds into `.grad` rather than overwriting it, so a training loop has to clear them before each new backward pass. This method is how.

## Where the network meets the engine

Nothing in `nn.py` imports a differentiation routine, an optimizer, or a tensor type. It only creates `Value`s and combines them with `+`, `*`, and `.relu()`. That means the two halves of a training step are already handled:

1. **Forward + backward.** Calling the model on some input builds the graph. Calling `.backward()` on the result fills in `.grad` for every parameter collected by `parameters()`.
2. **Update.** Gradient descent reads `p.data` and `p.grad` and nudges each parameter.

The generic loop looks like this, and it is the shape every example built on this repo uses:

```python
for epoch in range(EPOCHS):
    model.zero_grad()              # gradients accumulate, so clear them
    loss = loss_fn(model, X, y)    # forward: builds the graph
    loss.backward()                # backward: fills .grad for every parameter
    for p in model.parameters():
        p.data -= lr * p.grad      # vanilla SGD step
```

That is the entire training engine. Everything else — batching, learning-rate schedules, regularization, validation — is bookkeeping around these five lines.

## A worked check: one neuron by hand

Before trusting the whole network, it is worth checking a single neuron against arithmetic we can do ourselves. Let us build a two-input ReLU neuron and set its parameters to round numbers:

```python
from engine import Value
from nn import Neuron

n = Neuron(2)
n.w[0].data = 2.0
n.w[1].data = -1.0
n.b.data = 0.5

x = [Value(1.0), Value(2.0)]
out = n(x)
out.backward()

print(n)
print(f"out.data = {out.data}")
for i, p in enumerate(n.parameters()):
    print(f"p{i}: data={p.data: .4f} grad={p.grad: .4f}")
```

The forward value is

$$b + w_1 x_1 + w_2 x_2 = 0.5 + (2)(1) + (-1)(2) = 0.5,$$

and since the pre-activation is positive the ReLU leaves it alone. Running the code gives:

```
ReLUNeuron(2)
out.data = 0.5
p0: data= 2.0000 grad= 1.0000
p1: data=-1.0000 grad= 2.0000
p2: data= 0.5000 grad= 1.0000
```

These gradients are easy to verify by hand. The derivative of a ReLU at a positive input is $1$, so

$$\frac{\partial \text{out}}{\partial w_1} = x_1 = 1, \qquad
\frac{\partial \text{out}}{\partial w_2} = x_2 = 2, \qquad
\frac{\partial \text{out}}{\partial b} = 1.$$

The engine agrees: `p0.grad = 1`, `p1.grad = 2`, `p2.grad = 1`. No new gradient code was written for any of this — the `Value` nodes simply did their job.

## A worked check: a small MLP

Now the real thing. Let us build a `2 → 3 → 1` network, run one forward and backward pass, and look at the result:

```python
import random
from engine import Value
from nn import MLP

random.seed(1337)
model = MLP(2, [3, 1])
x = [Value(1.0), Value(2.0)]
y = model(x)
y.backward()

print(model)
print()
for i, p in enumerate(model.parameters()):
    print(f"p{i}: data={p.data: .4f} grad={p.grad: .4f}")
```

```
MLP of [
  0: Layer(2 -> 3) [3 x ReLUNeuron(2)]
  1: Layer(3 -> 1) [LinearNeuron(3)]
]

p0: data= 0.2355 grad=-0.2326
p1: data= 0.0665 grad=-0.4652
p2: data= 0.0000 grad=-0.2326
p3: data=-0.2683 grad= 0.5792
p4: data= 0.1716 grad= 1.1585
p5: data= 0.0000 grad= 0.5792
p6: data=-0.6686 grad= 0.8435
p7: data= 0.6487 grad= 1.6869
p8: data= 0.0000 grad= 0.8435
p9: data=-0.2326 grad= 0.3686
p10: data= 0.5792 grad= 0.0748
p11: data= 0.8435 grad= 0.6289
p12: data= 0.0000 grad= 1.0000
```

Reading this is a nice sanity check on the whole stack:

- `p0`–`p8` are the hidden layer: three neurons, each with two weights and a bias. `p9`–`p12` are the single output neuron (three weights and a bias).
- The output bias, `p12`, has gradient exactly `1.0`. That is `d(out)/d(out)` — the seed the engine sets before propagating backward.
- Each hidden bias gradient equals the matching output weight times that neuron's ReLU slope. For instance the first hidden neuron is active, so its bias gradient (`p2.grad = -0.2326`) equals the output weight that reads it (`p9.data = -0.2326`). When the weighted sum of a hidden neuron is negative its ReLU slope is `0`, and its gradients would collapse to zero — the network would learn nothing from it on that example.

A single forward-and-backward pass on thirteen parameters, and every gradient came out of the same generic loop from `engine.py`. The network classes only had to describe the arithmetic.

## Two things to keep in mind

- **The initialisation is deliberately simple.** Weights come from `random.uniform(-1, 1)` and biases start at zero. That is fine for the small problems here, but modern networks use variance-aware schemes such as He or Xavier initialisation, because a bad initial scale makes deep stacks train poorly or not at all.
- **The output layer is linear on purpose.** Because the last layer has `nonlin=False`, the network returns a logit. If you want probabilities you apply a sigmoid to the output afterwards; if you want to train, you use a loss that expects logits. Keeping the raw logit also avoids computing a sigmoid only to undo it in the loss.

## What's next

This file gives us a model we can differentiate, but not yet a problem to solve. The next article turns to `cancer_prediction.py`: loading the Breast Cancer Wisconsin dataset, training with binary cross-entropy *with logits*, and evaluating with stratified k-fold cross-validation — all using nothing but the `Value` engine and these three classes.
