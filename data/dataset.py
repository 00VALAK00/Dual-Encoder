import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from transformers import AutoTokenizer


class CustomDataset(Dataset):
    def __init__(self,tokenize=False):

        self.dataset = load_dataset("enelpol/rag-mini-bioasq", "question-answer-passages", split="train")
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        if tokenize:
            self.tokenize()

    def __len__(self):
        return len(self.dataset)

    def tokenize(self):
        self.dataset.set_format(type='torch', columns=['question', 'answer'])
        self.dataset = self.dataset.map(lambda exemple:
                                        {
                                            "question": self.tokenizer(exemple["question"], max_length=256,
                                                                       truncation=True, padding="max_length"),
                                            "answer": self.tokenizer(exemple["answer"], max_length=256,
                                                                     truncation=True, padding="max_length")
                                        })

    def __getitem__(self, index):

        question_dict = {k: v.squeeze(0) for k, v in self.dataset["question"][index].items()}
        answer_dict = {k: v.squeeze(0) for k, v in self.dataset["answer"][index].items()}
        return question_dict, answer_dict
