from typing import Dict, List, Tuple
from torch.utils.data import Dataset
import json

DEFAULT_LABELS = {
    "neutral": 0,
    "positive": 1,
    "negative": 2,
    "mixed": 3
}

class SentimentDataset(Dataset):

    def __init__(
        self, 
        records: List[dict], 
        label_to_idx: Dict[str, int]
    ) -> None:
        self.x = []
        self.y = []
        for rec in records:
            self.x.append(rec["text"])
            self.y.append(label_to_idx[rec["label"]])

    def __getitem__(self, idx) -> Tuple[str, int]:
        return self.x[idx], self.y[idx]

    def __len__(self) -> int:
        return len(self.y)
    
    @classmethod
    def from_jsonl(
        cls, 
        file: str, 
        label_to_idx: Dict[str, int] = DEFAULT_LABELS
    ):
        with open(file, "r", encoding="utf-8") as src:
            records = (json.loads(line) for line in src)
            return cls(records, label_to_idx)