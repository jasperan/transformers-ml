'''
Language modeling is a task that predicts a word in a sequence of text. It has become a very popular NLP task because a pretrained language model can be finetuned for many other downstream tasks. Lately, there has been a lot of interest in large language models (LLMs) which demonstrate zero- or few-shot learning. This means the model can solve tasks it wasn’t explicitly trained to do! Language models can be used to generate fluent and convincing text, though you need to be careful since the text may not always be accurate.

There are two types of language modeling:

    causal: the model’s objective is to predict the next token in a sequence, and future tokens are masked
    masked: the model’s objective is to predict a masked token in a sequence with full access to the tokens in the sequence    
'''

from transformers import pipeline

import warnings
warnings.filterwarnings("ignore")

prompt = "Hugging Face is a community-based open-source platform for machine learning."
generator = pipeline(task="text-generation")
print(generator(prompt))  # doctest: +SKIP