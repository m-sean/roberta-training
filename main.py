import argparse
import logging
import os
import torch

from loss import CrossEntropyLoss, FocalLoss, LabelSmoothingCrossEntropy
from model import RobertaSentimentModel
from optimizer import OptimizerConfig
from trainer import RobertaModelTrainer
    
def main(args: argparse.ArgumentParser) -> None:
    if not any([args.train, args.eval, args.trace]):
        raise ValueError("script requires one or more flags: `--train`, `--eval`, or `--trace`")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_trainer = RobertaModelTrainer(
        model_name=args.model_name,
        batch_size=args.batch_size, 
    )

    if args.train:
        train_file = f"{args.data_dir}/train.jsonl"
        train_loader = model_trainer.get_data_loader(train_file, train=True)
        dev_file = f"{args.data_dir}/dev.jsonl"
        val_loader = model_trainer.get_data_loader(dev_file, train=False)
        
        optimizer = OptimizerConfig.Adam
        optimizer.set_lr(args.lr)
        optimizer.set_betas(args.betas)
        optimizer.set_eps(args.eps)
        optimizer.set_decay(args.decay)
        optimizer.set_amsgrad(args.amsgrad)

        model_dir = args.model_dir if not args.model_dir.endswith("/") else args.model_dir[:-1]
        model_kwargs = {"dropout": args.dropout}
        model = model_trainer.train(
            train_loader,
            val_loader,
            loss=args.loss,
            epochs=args.epochs,
            save=model_dir,
            optimizer=optimizer,
            norm_clip=args.norm_clip,
            **model_kwargs
        )

    if args.eval:
        # load evaluation data
        test_file = f"{args.data_dir}/test.jsonl"
        test_loader = model_trainer.get_data_loader(test_file, train=False)
        # load the best model from training and evaluate
        model = RobertaSentimentModel.load(args.model_dir)
        model.eval()
        model.to(device)
        model_trainer.evaluate(model, test_loader)

    if args.trace:
        if not os.path.isdir(args.trace_dir):
            os.mkdir(args.trace_dir)
        file = f"{args.data_dir}/test.csv"
        loader = model_trainer.get_data_loader(file, train=True, shuffle=True)
        model = RobertaSentimentModel.load(args.model_dir)
        model.eval()
        model.to(device)
        x, _, mask = next(iter(loader))
        logits = model(x, mask)
        # save the model as a ScriptedModel that can be used in a rust env
        traced_model = torch.jit.trace(model, example_inputs=(x, mask))
        print(traced_model.code)
        trace_path = f"{args.trace_dir}/model.pt"
        torch.jit.save(traced_model, trace_path)
        logging.info(f"model trace saved to {trace_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=False, action="store_true")
    parser.add_argument("--eval", default=False, action="store_true")
    parser.add_argument("--trace", default=False, action="store_true")
    parser.add_argument("--data-dir", default="/tmp")
    parser.add_argument("--model-dir", default="roberta_sentiment")
    parser.add_argument("--trace-dir", default="roberta_trace")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--norm-clip", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    parser.add_argument("--decay", type=float, default=0.0)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--amsgrad", default=False, action="store_true")
    parser.add_argument("--model-name", choices=['roberta-base', 'roberta-large'], default='roberta-large')
    parser.add_argument("--loss", choices=['cce', 'focal', 'lsce'], default='lsce')
    logging.basicConfig(level="INFO", format="%(levelname)s: %(message)s")
    print(parser.parse_args())
    main(parser.parse_args())


