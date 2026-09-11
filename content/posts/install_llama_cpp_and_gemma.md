---
date: '2024-10-19T00:00:00-05:00'
draft: false
title: 'Install llama.cpp and Gemma from Huggingface'
author: 'Jorgen Bergstrom'
tags: ['LLM', 'Local LLM', 'Python']
categories: ['Local LLM']
---

## Installation
Learning how large language models (LLMs) like [ChatGPT](https://chatgpt.com) and [Gemini](https://gemini.google.com/app) work can be both fascinating and empowering. While using them through APIs is convenient, running one locally on your own computer unlocks deeper understanding and control. Fortunately, setting up your own LLM on Linux or Windows is surprisingly straightforward.
In this article, I will walk you through the simple steps to run a popular LLM like Transformers on your own computer. I will be using Linux as the operating system, but the process is equally applicable to Windows users. By going through these steps, you’ll gain valuable insight into how LLMs work under the hood and be able to customize their behavior for your specific needs.
```bash
## We will use llama.cpp to run a local LLM.
## The first step is to clone the llama.cpp repository from Github.
git clone https://github.com/ggerganov/llama.cpp.git
## Then make the llama files
cd llama.cpp
make
## Note that llama.cpp is simply the driver code for the Tranformer LLM
## We also need to download the weights for the LLM from the website Huggingface
## To do this we need to create an account on huggingface.co (this is free!)
## We also need to create and install a huggingface token. This is needed to get access to the LLMs.
## Once that is done I recommend getting the huggingface command line interface.
## This tool makes it easy to download different LLMs.
## The following command downloads the Gemma 7B model
huggingface-cli download google/gemma-1.1-7b-it
## Once the LLM has been downloaded, we need to convert it from hf to gguf file format.
## The llama.cpp repository has a tool for doing that. Just use the following command:
python convert_hf_to_gguf.py ~/.cache/huggingface/hub/models--google--gemma-1.1-7b-it/snaps
hots/065a528791af6f57f013e8e42b7276992b45ef71 --outfile gemma7.gguf --outtype auto
## That is it. We can now use the LLM. Here's the comand:
./llama-cli -m ~/.cache/huggingface/hub/models--google--gemma-1.1-7b-it/gemma.gguf --threads 8 -cnv --color
```
## Example of Running the LLM
Here is a screen recording showing the LLM being run from a command window. Being able to run this from your own computer (without any internet connection) is really cool!

<video width="100%" controls><source src="/Screen_Recording_Edit.mp4" type="video/mp4"></video>

## Running the GGUF model using Python
The following code can be used to run the pre-trained model using a few lines of Python.
```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-7b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-1.1-7b-it",
    torch_dtype=torch.bfloat16
)
input_text = "Write me a poem about Machine Learning."
input_ids = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**input_ids, max_new_tokens=50)
print(tokenizer.decode(outputs[0]))
```
