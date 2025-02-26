---
date: '2025-02-25T20:18:40-05:00'
draft: false
title: 'Set Up Python For ML'
author: 'Jorgen Bergstrom'
tags: ['ML', 'Python']
---


> This article explains how to set up a Python environment for use with Machine Learning

## Background

You can use any computer language for learning, coding, and solving  Machine Learning (ML) problems. The most common language to use is [Python](https://www.python.org/). The main reason for this is that there are many high quality libraries of ML algorithms already written and freely available for Python (e.g. [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [Keras](https://keras.io/)). This makes it really easy to get started and to try things out.

One main design feature of Python is that the core language is relatively small, and additional features are added through external packages. **This can lead to problems since you can only install a single version of a package at a time.**

The difficulty that can occur is that one package may depend on a specific version of another package, which can cause a conflict with some a third package. This problem is sometimes referred to as “dependency hell”.

The Python solution to this problem is to use a Python Virtual Environment. You can set up different virtual environments, and each virtual environment can have different packages and versions in it. This is quite handy and easy to do.

> **Run Python on Your Computer**
>
> If you are serious about learning ML, then I recommend that you develop and run your Python code on your own computer. This will give you more control, and it will prevent the hosting company from reusing your code. Also, it will allow you to learn more about computers and how to set up and run your own code. Which is very useful!