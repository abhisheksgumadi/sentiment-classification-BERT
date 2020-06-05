import pandas as pd
import transformers
from sklearn.model_selection import train_test_split
import torch
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import os
from torch import optim, nn
from transformers import (
	RobertaTokenizer,
	RobertaModel,
)
from collections import defaultdict
from CustomDataLoader import CustomDataLoader
from model import SentimentClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 

def create_data_loader(df, tokenizer, max_len, bs):
	ds = CustomDataLoader(
		df["text"].to_numpy(), df["score"].to_numpy(), tokenizer, max_len
	)

	return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=4)


def train_epoch(model, data_loader, loss_fn, optimizer, device, n_examples):
	model = model.train()
	print_every = 10  # prints the loss every 10 minibatches
	losses = []
	correct_predictions = 0
	num_batches = df_train.shape[0] // BATCH_SIZE
	for idx, d in enumerate(data_loader):
		input_ids = d["ids"].to(device)
		attention_mask = d["masks"].to(device)
		targets = d["scores"].to(device)

		outputs = model(input_ids=input_ids, attention_mask=attention_mask)

		_, preds = torch.max(outputs, dim=1)
		loss = loss_fn(outputs, targets)
		correct_predictions = correct_predictions + torch.sum(
			preds == targets
		)
		# print(type(correct_predictions.double()))
		loss_batch = loss.item()
		if (idx % print_every) == 0:
			print(f"The loss in {idx}th / {num_batches} batch is {loss_batch}")
		losses.append(loss_batch)

		loss.backward()
		nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
		optimizer.step()
		optimizer.zero_grad()

	return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
	model = model.eval()

	losses = []
	correct_predictions = 0

	with torch.no_grad():
		for d in data_loader:
			input_ids = d["ids"].to(device)
			attention_mask = d["masks"].to(device)
			targets = d["scores"].to(device)

			outputs = model(input_ids=input_ids, attention_mask=attention_mask)
			_, preds = torch.max(outputs, dim=1)

			loss = loss_fn(outputs, targets)

			correct_predictions += torch.sum(preds == targets)
			losses.append(loss.item())

	return correct_predictions.double() / n_examples, np.mean(losses)


def get_predictions(model, tokenizer, text, max_len):
	model = model.eval()
	encoded_output = tokenizer.encode_plus(
		text,
		add_special_tokens=True,
		max_length=max_len,
		pad_to_max_length=True,
		return_attention_mask=True,
		return_token_type_ids=True,
		return_tensors="pt",
	)
	ids = encoded_output["input_ids"]
	masks = encoded_output["attention_mask"]
	outputs = model(input_ids=ids, attention_mask=masks)
	_, preds = torch.max(outputs, dim=1)
	probs = F.softmax(outputs, dim=1)
	return preds.numpy(), probs.detach().numpy()


if __name__ == "__main__":
	config = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yml')))
	PRE_TRAINED_MODEL_NAME = "distilroberta-base"
	RANDOM_SEED = 1000

	BATCH_SIZE = config['batch_size']
	EPOCHS = config['num_epochs']
	NUM_CLASSES = config['num_classes']
	MAX_LEN = config['num_tokens']
	lr = config['lr']

	df = pd.read_csv(config['csv_path'])
	df_train, df_test = train_test_split(df, test_size=0.1, random_state=RANDOM_SEED)
	df_val, df_test = train_test_split(df_test, test_size=0.5, random_state=RANDOM_SEED)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	tokenizer = RobertaTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

	train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
	val_data_loader = create_data_loader(df_val, tokenizer, MAX_LEN, BATCH_SIZE)
	test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

	model = SentimentClassifier(NUM_CLASSES, PRE_TRAINED_MODEL_NAME)
	model = model.to(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	loss_fn = nn.CrossEntropyLoss().to(device)

	history = defaultdict(list)
	best_accuracy = 0

	for epoch in tqdm(range(EPOCHS)):
		print(f"Epoch {epoch + 1}/{EPOCHS}")
		print("-" * 10)

		train_acc, train_loss = train_epoch(
			model, train_data_loader, loss_fn, optimizer, device, len(df_train)
		)

		print(f"Epoch: {epoch}, Train loss: {train_loss}, accuracy: {train_acc}")

		val_acc, val_loss = eval_model(
			model, val_data_loader, loss_fn, device, len(df_val)
		)

		print(f"Epoch: {epoch}, Val loss: {val_loss}, accuracy: {val_acc}")

		history["train_acc"].append(train_acc)
		history["train_loss"].append(train_loss)
		history["val_acc"].append(val_acc)
		history["val_loss"].append(val_loss)

		if val_acc > best_accuracy:
			torch.save(model, "best_model_state.bin")
			best_accuracy = val_acc

	sample_text = "I really hate this movie"
	predictions, probabilities = get_predictions(model, tokenizer, sample_text, MAX_LEN)
	print(f'Text: {sample_text}, Prediction: {predictions[0]}, Probability: {np.max(probabilities[0])}')
