import torch

from trainer import RobertaModelTrainer


def main():
    model = torch.jit.load("roberta-lg-best/trace/model.pt", map_location="cpu")
    trainer = RobertaModelTrainer("roberta-large", batch_size=8)
    eval = trainer.get_data_loader("data/dev.jsonl", train=False)
    trainer.evaluate(model, eval, expanded_metrics=True)


if __name__ == "__main__":
    main()
