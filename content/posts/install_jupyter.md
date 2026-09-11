---
date: '2024-03-30T00:00:00-05:00'
draft: false
title: 'Install JupyterLab'
author: 'Jorgen Bergstrom'
tags: ['JupyterLab', 'Python']
categories: ['Python']
---

## Instructions

The next step after installing Python is to install [JupyterLab](https://jupyter.org/).
This is free open-source web-based interactive development environment for Python coding.
It is really cool, easy to install and use. All you have to do is to start Python and
type the following commands:
```
pip install --upgrade pip
pip install jupyterlab
```

![](/space_30.png)


Then start JupyterLab using the following command: `jupyter lab`.
This will open a new browser window with JupyterLab.  It will look something like in the following image.


{{< image src="/JupyterLab_Img-1024x545.webp" >}}


If you close the browser tab and want to open it again, then you can simply go to following url:
`http://localhost:8888/lab`.


If you need to at a later time, you can use the following command to update jupyterLab:
`pip install --upgrade jupyterlab`.
