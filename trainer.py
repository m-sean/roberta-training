import logging
from functools import partial
from tqdm import tqdm
from typing import Dict, Optional, Tuple

import convem
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from data import SentimentDataset, DEFAULT_LABELS
from loss import CrossEntropyLoss, FocalLoss, LabelSmoothingCrossEntropy
from model import RobertaSentimentModel
from optimizer import OptimizerConfig

def batch_collate_fn(batch, tokenizer: PreTrainedTokenizerFast, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x, y = zip(*batch)
    encodings = tokenizer(
        x,
        padding=True, 
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    x = encodings['input_ids'].to(device)
    mask = encodings['attention_mask'].to(device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return x, mask, y


def _get_loss_fn(loss: str, train: DataLoader, n_classes: int) -> torch.nn.Module:
    if loss == "lsce":
        return LabelSmoothingCrossEntropy()
    if loss == "focal":
        alphas=_get_alphas(train, n_classes)
        return FocalLoss(alpha=alphas, mean_reduce=True)
    return CrossEntropyLoss()

def _get_alphas(train: DataLoader, n_classes: int) -> torch.Tensor:
    counts = torch.zeros(n_classes)
    for (_, batch_labels, _) in train:
        for label in batch_labels.tolist():
            counts[label] += 1
    return counts / counts.sum()


class RobertaModelTrainer:

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        label_to_idx=DEFAULT_LABELS,
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.backend_tokenizer.enable_padding(pad_id=tokenizer.pad_token_id)
        tokenizer.backend_tokenizer.enable_truncation(max_length=tokenizer.model_max_length)
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_fn = None
        self.history = None
        self.label_to_idx = label_to_idx
        self.optimizer = None
        self.norm_clip = None
        self.collate_fn = partial(
            batch_collate_fn,
            tokenizer=tokenizer,
            device=self.device, 
        )

    def n_classes(self) -> int:
        return len(self.label_to_idx)

    def get_idx_to_label(self) -> Dict[int, str]:
        return {idx: label for label, idx in self.tag_to_idx.items()}

    def get_data_loader(self, data: str, train: bool) -> DataLoader:
        """Loads datasets from jsonl."""
        dataset = SentimentDataset.from_jsonl(data, self.label_to_idx)
        return DataLoader(dataset, self.batch_size, shuffle=train, collate_fn=self.collate_fn)
    
    def train(
        self,
        train: DataLoader,
        val: DataLoader,
        loss: str,
        epochs: int,
        save: str,
        optimizer: OptimizerConfig,
        norm_clip: Optional[float] = None,
        **model_kwargs,
    ) -> RobertaSentimentModel:
        """Performs model training with early stopping after no improvement in validation after 3 epochs.
        
        Args:
            train: Training set Dataloader.
            val: Validation set Dataloader.
            save: Location to save the best model from training.
            optimizer: Optimizer configuration.
            model_kwargs: Model constructor args
        """
        n_classes = self.n_classes()
        self.norm_clip = norm_clip
        self.loss_fn = _get_loss_fn(loss, train, n_classes).to(self.device)
        model = RobertaSentimentModel(
            model_name=self.model_name, 
            n_classes=n_classes,
            **model_kwargs
        ).to(device=self.device).train()
        self.optimizer = optimizer.get_optimizer(model.trainable_params())
        self.history = {"train": [], "val": []}
        batch_ct = len(train)
        for e in range(1, epochs + 1):
            with tqdm(total=batch_ct, unit="batch") as pbar:
                pbar.set_description(f"Epoch {e}")
                t_acc, t_rk, t_loss = self._train_step(model, train, pbar)
                v_acc, v_rk, v_loss = self._val_step(model, val, pbar)
                pbar.set_postfix_str(
                    f"t_acc={t_acc:.3f} t_rk={t_rk:.3f} t_loss={t_loss:.3f} "
                    f"v_acc={v_acc:.3f} v_rk={v_rk:.3f} v_loss={v_loss:.3f}"
                )
            self.history["train"].append((t_acc, t_rk, t_loss))
            self.history["val"].append((v_acc, t_rk, v_loss))
            if e == 1 or v_rk > best_rk:
                logging.info(f"Saving model to {save}/model.pt")
                best_epoch = e
                best_rk = v_rk
                model.save(save)
            elif e - best_epoch > 4:
                logging.info("No improvement after 3 epochs, stopping early.")
                break
        return model
    
    def evaluate(self, model: RobertaSentimentModel, eval: DataLoader) -> Tuple[float, float, float]:
        logging.info("Running model evaluation")
        batch_ct = len(eval)
        with tqdm(total=batch_ct, unit="batch") as pbar:
            pbar.set_description("Model evaluation")
            acc, rk, loss = self._val_step(model, eval, pbar)
            pbar.set_postfix_str(
                f"ACC={acc:.3f} "
                f"RK={rk:.3f} "
                f"LOSS={loss:.3f}"
            )
        return (acc, rk, loss)

    def _train_step(
        self, model: RobertaSentimentModel, train: DataLoader, pbar: tqdm
    ) -> Tuple[float, float, float]:
        sum_acc = 0.0
        sum_loss = 0.0
        sum_rk = 0.0
        model.train()
        for i, (x, mask, labels) in enumerate(train):
            logits = model(input_ids=x, attention_mask=mask)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            if self.norm_clip:
                clip_grad_norm_(model.parameters(), self.norm_clip)
            self.optimizer.step()
            model.zero_grad()
            mcm = convem.multi_confusion_matrix(logits.tolist(), labels.tolist())
            sum_acc += mcm.accuracy()
            sum_rk += mcm.rk() or 0.0
            sum_loss += loss.detach()
            n = i + 1
            avg_acc = sum_acc / n
            avg_rk = sum_rk / n
            avg_loss = sum_loss / n
            pbar.update()
            pbar.set_postfix_str(f"acc={avg_acc:.3f} rk={avg_rk:.3f} loss={avg_loss:.3f}")
        return avg_acc, avg_rk, avg_loss

    def _val_step(self, model: RobertaSentimentModel, val: DataLoader, pbar: tqdm) -> Tuple[float, float, float]:
        all_labels = []
        all_predicitions = []
        sum_loss = 0.0
        with torch.no_grad():
            model.eval()
            for i, (x, mask, labels) in enumerate(val):
                logits = model(input_ids=x, attention_mask=mask)
                sum_loss += self.loss_fn(logits, labels)
                all_labels.extend(labels.tolist())
                all_predicitions.extend(logits.tolist())
                pbar.update()
        mcm = convem.multi_confusion_matrix(all_predicitions, all_labels)
        return mcm.accuracy(), mcm.rk(), sum_loss / (i + 1)
