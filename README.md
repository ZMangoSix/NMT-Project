# NMT-Project
Neural Machine Translation (NMT) project for EECS 595 Course.

## What is NMT (Neural Machine Translation)
Machine translation represents a cornerstone in the field of Natural Language Processing (NLP), embodying the sophisticated endeavor of translating text or speech from one language to another. This process not only requires accurate transposition of words but also demands a deep understanding of the contextual, idiomatic, and pragmatic nuances inherent to each language. Historically, machine translation has navigated through various phases, initially leveraging rule-based and probabilistic machine learning approaches to surmount the linguistic barriers. However, the recent surge in machine learning advancements has ushered in the era of neural networks, revolutionizing machine translation with their capacity to deliver remarkably superior performance. This project delves into the realm of neural machine translation (NMT), a testament to the strides made in NLP, showcasing its potential as both a formidable tool in future industrial opportunities and a foundational pilot study for expansive research endeavors. Through this exploration, we aim to not only highlight the technical prowess of current NMT systems but also to pave the way for further innovations in overcoming the challenges of cross-lingual communication.

## About this Repo
The project is based on the python (3.9.15) and Pytorch (>=2.0.1).

Hugging Face is also used for text tokenization. To use the related function, one has to install the following packages
```
pip install transformers sentencepiece sacremoses evaluate
```
The encoder-decoder architecture is used for machine translation tasks.

Different ML algorithms are tried to achieve a better performance and also for us to understand their differences.
- RNN-based models like LSTM, GRU
- Transformer model

## How to run the Repo
Just go to the jupyter notebook NMT.ipynb. follow the instructions inside the notebook. All used data for the en-fr  and en-zh translation models are stored in the data folder.
You can have your own dataset, just need to modify the dataset.py file to read them.
