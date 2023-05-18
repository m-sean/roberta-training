from typing import Dict, Optional
import os
import json
import torch
import torch.nn as nn
from transformers import RobertaModel

class RobertaSentimentModel(nn.Module):

    def __init__(self, model_name: str, n_classes: int, dropout: Optional[float] = 0.0) -> None:
        self.config = {
            "model_name": model_name,
            "n_classes": n_classes,
            "dropout": dropout,
        }
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(model_name)
        seq_dim = self.roberta.config.max_position_embeddings
        emb_dim = self.roberta.config.hidden_size
        self.fc = nn.Linear(emb_dim, seq_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(seq_dim, n_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        x = self.roberta(input_ids, attention_mask=attention_mask, return_dict=False)[0]
        x = self.fc(x)
        x = self.relu(x)
        x = x * torch.unsqueeze(attention_mask, 2)
        x = torch.sum(x, 1) / torch.unsqueeze(torch.sum(attention_mask, 1), 1)
        x = self.classifier(x)
        return x
    
    def trainable_params(self):
        return (param for _, param in self.named_parameters())
    
    def _save_config(self, dir):
        with open(f"{dir}/config.json", "w") as fp:
            json.dump(self.config, fp)

    @staticmethod
    def _load_config(dir) -> Dict[str, any]:
        with open(f"{dir}/config.json", "r") as fp:
            return json.load(fp)
            
    def save(self, dir: str) -> None:
        if not os.path.isdir(dir):
            os.mkdir(dir)
        self._save_config(dir)
        torch.save(self.state_dict(), f=f"{dir}/model.pt")

    @classmethod
    def load(cls, dir: str):
        config = cls._load_config(dir)
        model = cls(**config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_dict = torch.load(f"{dir}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        return model