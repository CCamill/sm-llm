import torch
from torch.utils.data import Dataset, DataLoader, random_split
from dataload import FunctionSliceDataset
from transformers import AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments
from transformers import DataCollatorForLanguageModeling

class Fintune():
    def __init__(self, train_dataloader, val_dataloader):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.model_name = "microsoft/codebert-base"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
    def train(self):
        
if __name__ == '__main__':
    data = FunctionSliceDataset(root_dir='resources/datasets/source_funcs_treesitter')
    train_size = int(0.8*len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = random_split(data, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
