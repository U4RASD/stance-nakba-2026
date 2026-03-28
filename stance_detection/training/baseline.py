"""Baseline training (cross-entropy) and optional stratified CV."""

import argparse

from transformers import AutoModelForSequenceClassification, Trainer

from ..config import TrainConfig
from ..data_loader import StanceDataset
from .cv import (
    FoldContext, CVConfig, CrossValidator,
    cv_compute_metrics, evaluate_fold, make_fold_training_args,
)
from .utils import (
    add_common_args, make_config, compute_metrics,
    run_single_phase,
)


def _make_trainer(model, training_args, train_dataset, eval_dataset):
    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


def run_cv_fold(ctx: FoldContext):
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

    training_args = make_fold_training_args(ctx)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=cv_compute_metrics,
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
        description="Train baseline stance model (CE)")
    add_common_args(parser)
    args = parser.parse_args()

    if args.cross_validate:
        config = make_config(args)
        if not args.output_dir:
            config.output_dir = f"./outputs/cv_basic_{args.n_folds}fold"
        cv_config = CVConfig(n_folds=args.n_folds,
                             save_all_folds=args.save_all_folds)
        cv = CrossValidator(
            train_config=config,
            cv_config=cv_config,
            fold_runner_fn=run_cv_fold,
            method_name="basic",
            train_path=getattr(args, "train_path", None),
        )
        results = cv.run()
        print(f"\nMacro F1: {results['macro_f1_mean']:.4f} "
              f"+/- {results['macro_f1_std']:.4f}")
        return

    run_single_phase(args, trainer_factory=_make_trainer,
                     method_label="Baseline")


if __name__ == "__main__":
    main()
