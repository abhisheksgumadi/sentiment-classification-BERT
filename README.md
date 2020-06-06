# Fine Tuning BERT for Sentiment Analysis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository shows how to fine tune a BERT like model from HuggingFace for Sentiment classification. The model we consider if a RoBERTa model from HuggingFace. The code can be modified easily to support any other BERT kind of a model for fine tuning. I have used multiple references around the internet to come up with this single piece of code as a tutorial for anyone familiarizing themselves with HuggingFace and PyTorch. I also recommend reading the original blog that helped me compile the code from [here](https://www.curiousily.com/posts/sentiment-analysis-with-bert-and-hugging-face-using-pytorch-and-python/)

## DEPENDENCIES:
* torch == 1.5.0
* pandas == 1.0.4
* sklearn == 0.23.1
* PyYAML == 5.3.1
* transformers == 2.11.0

You can even create a Pipenv virtual environment using the Pipfile provided by running

```python
pipenv install .
```
  
## NOTE: 

The objective of this repository is **NOT** to provide a state of the art sentiment classification model. This is **NOT** production level code. Instead the goal is to provide you a tutorial on how to train a model in PyTorch using HuggingFace library :relaxed:

## INPUT DATA

For the impatient :smile: , sample data is in a file named `sample_data.csv`. You can note down the format in this file and straight skip to the **USAGE** section below on how to run the code. 

It is very easy to create an input dataset for training. All you need is Pandas DataFrame with two column viz. **"text"** and **"score"**. Once you create a Pandas Dataframe in this format where the text column contains the raw text and the **"score"** columns contains the sentiment score of the text, just save it as a csv file and mention the path of the file in the `config.yml` file as mentioned below. Note that the scores are ordinal numbers like `0`, `1` and so on starting with `0`. You can internally maintain a correspondance within your code on what classes `0`, `1` means and so on.

## CONFIG FILE

To run the code, you will first need to edit a config file for the following parameters. The config file is `config.yml` file found within the `src` directory.

* `num_epochs`: The number of epochs you want the code to train
* `num_tokens`: The maximum number of tokens you want to consider in the input during training for the BERT model
* `batch_size`: The size of the mini batch during training
* `num_classes`: The number of classes for classification
* `lr`: The learning rate for training
* `csv_path`: The path to the csv file with the input structure as defined above

## INSTALLATION & USAGE

Clone the repo 

```bash
git clone https://github.com/abhisheksgumadi/sentiment-classification-BERT.git
cd sentiment-classification-BERT
```
Once you are inside the main directory, run
```python
src/train.py
``` 

that automatically picks up the configuration inside the `src/config.yml` file and starts training based on the provided configuration parameters.

Have fun! and play around with the code to understand how to train a text classification model by fine tuning BERT using HuggingFace :blush:. Change the codebase hovewer you want, experiment with it and most importantly learn something new today :smile:
