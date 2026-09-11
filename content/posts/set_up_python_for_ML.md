---
date: '2024-03-17T00:00:00-05:00'
draft: false
title: 'Set Up Python For ML'
author: 'Jorgen Bergstrom'
tags: ['ML', 'Python']
---

## Background

You can use any computer language for learning, coding, and solving  Machine Learning (ML) problems. The most common language to use is [Python](https://www.python.org/). The main reason for this is that there are many high quality libraries of ML algorithms already written and freely available for Python (e.g. [PyTorch](https://pytorch.org/), [TensorFlow](https://www.tensorflow.org/), and [Keras](https://keras.io/)). This makes it really easy to get started and to try things out.

One main design feature of Python is that the core language is relatively small, and additional features are added through external packages. **This can lead to problems since you can only install a single version of a package at a time.**

The difficulty that can occur is that one package may depend on a specific version of another package, which can cause a conflict with some a third package. This problem is sometimes referred to as “dependency hell”.

The Python solution to this problem is to use a Python Virtual Environment. You can set up different virtual environments, and each virtual environment can have different packages and versions in it. This is quite handy and easy to do.

{{< admonition type=note title="Run Python on Your Computer" >}}
If you are serious about learning ML, then I recommend that you develop and run your Python code on your own computer. This will give you more control, and it will prevent the hosting company from reusing your code. Also, it will allow you to learn more about computers and how to set up and run your own code. Which is very useful!
{{< /admonition >}}

{{< admonition type=note title="Run Python on a Hosted Computer" >}}
The quickest and easiest way to run Python is to use a hosted Jupiter Notebook (for example, [Google Colab](https://colab.google/)). This can be very handy, but if you are doing anything serious then I would recommend using your own computer so you can control access to the code.
{{< /admonition >}}

![](/space_30.png)

{{< image src="/linux_icon_512x512.png" width="64" >}}
## Installing Python and a Virtual Environment on a Linux Computer

In this section I will show how you can install Python and set up a virtual environment on a Ubuntu-based Linux computer. Run the following terminal commands one time to set it up.

```
sudo apt install python3 python3-pip python3-venv
python3 -m venv ~/Documents/MegaML_Python
```

Then run the following command to activate the python environment. You should repeat this command every time you start a new terminal window.

```
source ~/Documents/MegaML_Python/bin/activate
```

![](/space_30.png)

{{< image src="/windows_icon_512x512.png" width="64" >}}
## Installing Python and a Virtual Environment on a Windows

One of the easiest ways to install Python on a Windows computer is to download Python from: [https://www.python.org/downloads/](https://www.python.org/downloads/). Then run the same command to set up and activate the Virtual Environment as for Linux. That is it!
