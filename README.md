# sentiment-classification-BERT.  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository show how to fine tune a BERT like model from HuggingFace for Sentiment classification. The model we consider if a RoBERTa model from HuggingFace. The code can be modified easily to support any other BERT kind of a model for fine tuning. 

## NOTE: The objective of this repository is **NOT** to provide you with a state of the art sentiment classification model :wink:. Instead the goal is to show you how to train one in PyTorch using HuggingFace library :relaxed:

## INPUT DATA

It is very easy to create an input dataset for training. All you need is Pandas DataFrame with two column viz. **"text"** and **"score"**. Once you create a Pandas Dataframe in this format where the text column contains the raw text and the **"score"** columns contains the sentiment score of the text, just save it as a csv file and mention the path of the file in the `config.yml` file as mentioned below. Note that the scores are ordinal numbers like `0`, `1` and so on starting with `0`. You can internally maintain a correspondance within your code on what classes `0`, `1` means and so on.

## CONFIG FILE

To run the code, you will first need to edit a config file for the following parameters. The config file is `config.yml` file found within the `src` directory.

* `num_epochs`: The number of epochs you want the code to train
* `num_tokens`: The maximum number of tokens you want to consider in the input during training for the BERT model
* `batch_size`: The size of the mini batch during training
* `num_classes`: The number of classes for classification
* `lr`: The learning rate for training
* `csv_path`: The path to the csv file with the input structure as defined above

## USAGE

To start training, run the `src/train.py` file that automatically picks up the configuration inside the `src/config.yml` file. 

Have fun! and play around with the code to understand how to train a text classification model by fine tuning BERT using HuggingFace :blush:

For any questions, raise an issue :smirk: or email me at `abhisheksgumadi@gmail.com`
