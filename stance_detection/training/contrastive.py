"""CE plus contrastive loss (pytorch_metric_learning); single run or CV."""

import argparse
from functools import partial

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, Trainer

from ..config import TrainConfig
from ..data_loader import StanceDataset
from .cv import (
    FoldContext, CVConfig, CrossValidator,
    cv_compute_metrics, evaluate_fold, make_fold_training_args,
)
from .utils import (
    add_common_args, make_config, set_seed, compute_metrics,
    build_model, run_single_phase,
)


LOSS_CHOICES = [
    "supcon", "contrastive", "tuplet", "ntxent", "triplet", "multisimilarity",
    "circle", "cosface", "arcface",
]


def build_contrastive_loss(loss_type: str, temperature: float = 0.07,
                           margin: float = 0.05, num_classes: int = None,
                           embedding_size: int = None):
    from pytorch_metric_learning import losses

    extra_params = []

    if loss_type == "supcon":
        loss_fn = losses.SupConLoss()
    elif loss_type == "contrastive":
        loss_fn = losses.ContrastiveLoss()
    elif loss_type == "tuplet":
        loss_fn = losses.TupletMarginLoss()
    elif loss_type == "ntxent":
        loss_fn = losses.NTXentLoss()
    elif loss_type == "triplet":
        loss_fn = losses.TripletMarginLoss()
    elif loss_type == "multisimilarity":
        loss_fn = losses.MultiSimilarityLoss()
    elif loss_type == "circle":
        loss_fn = losses.CircleLoss()
    elif loss_type == "cosface":
        assert num_classes and embedding_size, (
            "cosface requires num_classes and embedding_size")
        loss_fn = losses.CosFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
        )
        extra_params = list(loss_fn.parameters())
    elif loss_type == "arcface":
        assert num_classes and embedding_size, (
            "arcface requires num_classes and embedding_size")
        loss_fn = losses.ArcFaceLoss(
            num_classes=num_classes,
            embedding_size=embedding_size,
        )
        extra_params = list(loss_fn.parameters())
    else:
        raise ValueError(f"Unknown contrastive loss: {loss_type!r}. "
                         f"Choose from {LOSS_CHOICES}")

    return loss_fn, extra_params


class ContrastiveTrainer(Trainer):
    """Trainer combining Cross Entropy with a contrastive loss.

    The contrastive loss operates on L2-normalised CLS embeddings and ground
    truth labels.  Any pytorch_metric_learning loss that follows the standard
    (embeddings, labels) calling convention is supported.
    """

    def __init__(self, *args, contrastive_weight: float = 0.1,
                 contrastive_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrastive_weight = contrastive_weight
        self.contrastive_loss_fn = contrastive_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False,
                     num_items_in_batch=None):
        labels = inputs.get("labels")

        outputs = model(**inputs, output_hidden_states=True)
        ce_loss = outputs.loss

        hidden_states = outputs.hidden_states[-1]
        cls_embeddings = hidden_states[:, 0, :]
        cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)

        if labels is not None and cls_embeddings.size(0) > 1:
            contrastive_loss = self.contrastive_loss_fn(cls_embeddings, labels)
        else:
            contrastive_loss = torch.tensor(0.0, device=cls_embeddings.device)

        total_loss = ce_loss + self.contrastive_weight * contrastive_loss

        return (total_loss, outputs) if return_outputs else total_loss


def _make_trainer_factory(contrastive_weight: float, loss_type: str,
                          temperature: float, margin: float):
    """Return a trainer factory pre-configured with contrastive hyperparameters."""
    def factory(model, training_args, train_dataset, eval_dataset):
        num_classes = model.config.num_labels
        embedding_size = model.config.hidden_size
        loss_fn, extra_params = build_contrastive_loss(
            loss_type, temperature=temperature, margin=margin,
            num_classes=num_classes, embedding_size=embedding_size,
        )
        loss_fn = loss_fn.to(model.device)

        trainer = ContrastiveTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            contrastive_weight=contrastive_weight,
            contrastive_loss_fn=loss_fn,
        )
        if extra_params:
            trainer._contrastive_extra_params = extra_params
        return trainer
    return factory


def _make_cv_loss_fn(loss_type, temperature, margin, num_labels, hidden_size,
                     device):
    """Build and move a contrastive loss to the right device for a CV fold."""
    loss_fn, _ = build_contrastive_loss(
        loss_type, temperature=temperature, margin=margin,
        num_classes=num_labels, embedding_size=hidden_size,
    )
    return loss_fn.to(device)


def run_cv_fold(ctx: FoldContext, contrastive_weight=0.1,
                loss_type="supcon", temperature=0.07, margin=0.05):
    train_dataset = StanceDataset(
        texts=ctx.fold_train_df[ctx.text_col].tolist(),
        topics=ctx.fold_train_df[ctx.topic_col].tolist() if ctx.topic_col else None,
        labels=[ctx.label2id[l] for l in ctx.fold_train_df[ctx.label_col]],
        tokenizer=ctx.tokenizer,
        max_length=ctx.train_config.max_length,
    )
    val_dataset = StanceDataset(
        texts=ctx.fold_val_df[ctx.text_col].tolist(),
        topics=ctx.fold_val_df[ctx.topic_col].tolist() if ctx.topic_col else None,
        labels=[ctx.label2id[l] for l in ctx.fold_val_df[ctx.label_col]],
        tokenizer=ctx.tokenizer,
        max_length=ctx.train_config.max_length,
    )

    model_kwargs = dict(
        num_labels=ctx.num_labels,
        id2label=ctx.id2label,
        label2id=ctx.label2id,
        trust_remote_code=True,
    )
    if ctx.train_config.classifier_dropout is not None:
        model_kwargs["classifier_dropout"] = ctx.train_config.classifier_dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        ctx.train_config.model_name, **model_kwargs,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loss_fn = _make_cv_loss_fn(
        loss_type, temperature, margin,
        ctx.num_labels, model.config.hidden_size, device,
    )

    training_args = make_fold_training_args(ctx)
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=cv_compute_metrics,
        contrastive_weight=contrastive_weight,
        contrastive_loss_fn=loss_fn,
    )

    trainer.train()
    metrics = evaluate_fold(trainer, val_dataset, ctx.id2label, ctx.num_labels,
                            fold_val_df=ctx.fold_val_df, topic_col=ctx.topic_col)

    if ctx.save_fold_model:
        trainer.save_model(ctx.output_dir)
        ctx.tokenizer.save_pretrained(ctx.output_dir)

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Train with CE + contrastive loss")
    add_common_args(parser)
    parser.add_argument("-cw", "--contrastive-weight", type=float, default=0.1,
                        help="Contrastive loss weight")
    parser.add_argument("-cl", "--loss", type=str, default="supcon",
                        choices=LOSS_CHOICES,
                        help="Contrastive loss type")
    parser.add_argument("-t", "--temperature", type=float, default=0.07,
                        help="Temperature for supcon/ntxent")
    parser.add_argument("--margin", type=float, default=0.05,
                        help="Margin for triplet/circle/arcface/cosface")
    args = parser.parse_args()

    print(f"[Contrastive Training]  loss={args.loss}  "
          f"weight={args.contrastive_weight}  temperature={args.temperature}  "
          f"margin={args.margin}")

    trainer_factory = _make_trainer_factory(
        args.contrastive_weight, args.loss, args.temperature, args.margin,
    )

    if args.cross_validate:
        config = make_config(args)
        if not args.output_dir:
            config.output_dir = (
                f"./outputs/cv_contrastive_{args.loss}_{args.n_folds}fold")
        cv_config = CVConfig(n_folds=args.n_folds,
                             save_all_folds=args.save_all_folds)
        fold_runner = partial(run_cv_fold,
                              contrastive_weight=args.contrastive_weight,
                              loss_type=args.loss,
                              temperature=args.temperature,
                              margin=args.margin)
        cv = CrossValidator(
            train_config=config,
            cv_config=cv_config,
            fold_runner_fn=fold_runner,
            method_name=f"contrastive_{args.loss}",
            train_path=getattr(args, "train_path", None),
        )
        results = cv.run()
        print(f"\nMacro F1: {results['macro_f1_mean']:.4f} "
              f"+/- {results['macro_f1_std']:.4f}")
        return

    method_label = f"CE + Contrastive ({args.loss})"
    run_single_phase(args, trainer_factory=trainer_factory,
                     method_label=method_label)


if __name__ == "__main__":
    main()
