from torch.utils.data import Dataset, DataLoader
import torch


class CustomDataLoader(Dataset):
    def __init__(self, reviews, sent_scores, tokenizer, max_len):
        self.reviews = reviews
        self.scores = sent_scores
        self.max_len = max_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, item_idx):
        review = str(self.reviews[item_idx])
        score = self.scores[item_idx]
        encoded_output = self.tokenizer.encode_plus(
            review,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors="pt",
        )

        return {
            "texts": review,
            "ids": encoded_output["input_ids"].flatten(),
            "masks": encoded_output["attention_mask"].flatten(),
            "scores": torch.tensor(score, dtype=torch.long),
        }
