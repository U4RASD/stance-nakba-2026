"""CLI entry: multitask stance + sarcasm + sentiment; single run or CV."""

import argparse

from .cv import (
    FoldContext, CVConfig, CrossValidator,
    cv_compute_metrics, evaluate_fold, make_fold_training_args,
)
from .utils import add_common_args, make_config, compute_metrics
from .multitask import (
    MultitaskDataset, MultitaskTrainer, load_multitask_model,
    label_dataset, train_multitask,
    predict_multitask,
)


def _make_trainer(model, training_args, train_dataset, eval_dataset):
    return MultitaskTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )


def run_cv_fold(ctx: FoldContext):
    aux = ctx.prepared_data
    train_sarcasm = [aux["sarcasm_labels"][i] for i in ctx.train_idx]
    train_sentiment = [aux["sentiment_labels"][i] for i in ctx.train_idx]
    val_sarcasm = [aux["sarcasm_labels"][i] for i in ctx.val_idx]
    val_sentiment = [aux["sentiment_labels"][i] for i in ctx.val_idx]

    num_sarcasm = len(set(aux["sarcasm_labels"]))
    num_sentiment = len(set(aux["sentiment_labels"]))

    train_dataset = MultitaskDataset(
        ctx.fold_train_df[ctx.text_col].tolist(),
        ctx.fold_train_df[ctx.topic_col].tolist() if ctx.topic_col else None,
        [ctx.label2id[l] for l in ctx.fold_train_df[ctx.label_col]],
        train_sarcasm, train_sentiment, ctx.tokenizer, ctx.train_config.max_length,
    )
    val_dataset = MultitaskDataset(
        ctx.fold_val_df[ctx.text_col].tolist(),
        ctx.fold_val_df[ctx.topic_col].tolist() if ctx.topic_col else None,
        [ctx.label2id[l] for l in ctx.fold_val_df[ctx.label_col]],
        val_sarcasm, val_sentiment, ctx.tokenizer, ctx.train_config.max_length,
    )

    model = load_multitask_model(ctx.train_config.model_name, ctx.num_labels,
                                 num_sarcasm, num_sentiment,
                                 classifier_dropout=ctx.train_config.classifier_dropout)
    training_args = make_fold_training_args(ctx, remove_unused_columns=False)
    trainer = MultitaskTrainer(
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
        description="Train multitask (stance + sarcasm + sentiment)")
    add_common_args(parser)
    parser.add_argument("-nc", "--no-cache", action="store_true",
                        help="Re-label auxiliary data (ignore cache)")
    args = parser.parse_args()

    if args.cross_validate:
        config = make_config(args)
        if not args.output_dir:
            config.output_dir = f"./outputs/cv_multitask_{args.n_folds}fold"
        use_cache = not args.no_cache
        prepare_fn = lambda df, sc: label_dataset(
            df, sc["text_col"], use_cache, args.subtask, args.train_path)
        cv = CrossValidator(
            train_config=config,
            cv_config=CVConfig(n_folds=args.n_folds,
                               save_all_folds=args.save_all_folds),
            fold_runner_fn=run_cv_fold,
            method_name="multitask",
            train_path=args.train_path,
            prepare_fn=prepare_fn,
        )
        results = cv.run()
        print(f"\nMacro F1: {results['macro_f1_mean']:.4f} "
              f"+/- {results['macro_f1_std']:.4f}")
        return

    config = make_config(args)
    trainer, data = train_multitask(
        config, _make_trainer,
        use_cache=not args.no_cache,
        arabert_model=args.arabert_prep,
        train_path=args.train_path,
        val_path=args.val_path,
    )
    predict_multitask(trainer, data, config)


if __name__ == "__main__":
    main()
