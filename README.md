# Arabic Stance Detection - StanceNakba 2026

Code for the U4RASD system paper on Arabic stance detection in the political domain as part of the StanceNakba 2026 shared task.

## Setup

```bash
pip install -e .
```

For LLM experiments, copy the env template and fill in your API key:

```bash
cp stance_detection/.env.example stance_detection/.env
```

## Training

All training scripts share common CLI flags (run with `--help`).

```bash
# Baseline (cross-entropy)
python -m stance_detection.training.baseline -st B -m UBC-NLP/MARBERTv2

# CE + contrastive loss
python -m stance_detection.training.contrastive -st B -m UBC-NLP/MARBERTv2 --loss contrastive

# Multitask (stance + sarcasm + sentiment)
python -m stance_detection.training.multitask_train -st B -m UBC-NLP/MARBERTv2

# Any of the above with cross-validation
python -m stance_detection.training.baseline -st B -m UBC-NLP/MARBERTv2 --cross-validate --n-folds 5
```

## LLM Predictions

```bash
python -m stance_detection.llm.predict -st B -s val --task stance
```

## Clustering

Centroid-based classifier that embeds documents and assigns stances by nearest cluster centroid.

```bash
# Train from a topic/stance/*.txt folder structure
python -m stance_detection.clustering.train ./data -o model.pkl -m Qwen/Qwen3-Embedding-8B

# Evaluate on a labelled CSV
python -m stance_detection.clustering.evaluate dataset.csv model.pkl -o ./eval_results

# Predict a single text
python -m stance_detection.clustering.predict model.pkl topic_name "text to classify"
```

## Evaluation

```bash
# Evaluate a saved model on a labeled CSV
python -m stance_detection.inference.evaluate_model -m outputs/model -d data.csv

# Compare two prediction CSVs
python -m stance_detection.inference.eval -p predictions.csv -t ground_truth.csv
```

## Project Structure

```
stance_detection/
    config.py               Training config dataclass
    data_loader.py          Data loading, normalization, StanceDataset
    training/
        baseline.py         CE baseline
        contrastive.py      CE + contrastive loss
        multitask.py        Multitask model and training logic
        multitask_train.py  Multitask CLI entry point
        cv.py               Stratified K-fold cross-validation
        utils.py            Shared training helpers
    clustering/
        classifier.py       Nearest-centroid stance classifier
        cluster.py          Cluster management and distance metrics
        embedder.py         HuggingFace embedding wrapper
        data_loader.py      Folder-based document loader
        train.py            Training CLI
        evaluate.py         Evaluation CLI
        predict.py          Prediction CLI
    llm/
        client.py           OpenAI compatible LLM client
        prompting.py        Prompts and response schemas
        predict.py          Batch LLM prediction pipeline
    inference/
        eval.py             Prediction vs ground-truth evaluation
        evaluate_model.py   End-to-end model inference and eval
    tools/
        combine.py          Merge LLM auxiliary labels into CSVs
        combine_augmented.py  Combine original + augmented data
        effective_rank.py   RankMe comparison for two models
```
