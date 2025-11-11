import logging
import argparse
import os
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, RobertaConfig, TrainerCallback
from datasets import load_dataset
import evaluate
import numpy as np
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    # Note: The below default values are the default values of the Trainer class, which is the commonly used values.
    # For baseline with no regularization and optimization techniques update the values to the following:
    # --override-dropout = True
    # --dropout = 0.0
    # --weight-decay = 0.0
    # --optimizer = 'sgd'
    
    parser = argparse.ArgumentParser(description='Train RoBERTa on AG News with feature flags')
    parser.add_argument('--override-dropout', action='store_true', help='Override dropout via config')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability to set when overriding')
    parser.add_argument('--use-early-stopping', action='store_true', help='Enable early stopping callback')
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--train-batch-size', type=int, default=8, help='Per-device train batch size')
    parser.add_argument('--eval-batch-size', type=int, default=8, help='Per-device eval batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--optimizer', default='adamw_torch', help='Optimizer type, for valid values refer to OptimizerNames in https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py')
    return parser.parse_args()

def main(args):
    # Load AG News dataset
    logger.info("Loading AG News dataset")
    dataset = load_dataset("ag_news")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    logger.info("Creating validation split from train set (20%)")
    split = train_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split['train']
    val_dataset = split['test']

    # Load the tokenizer and model
    logger.info("Loading tokenizer and model")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    if args.override_dropout:
        logger.info(f"Overriding dropout in config to {args.dropout}")
        config = RobertaConfig.from_pretrained(
            'roberta-base',
            hidden_dropout_prob=args.dropout,
            attention_probs_dropout_prob=args.dropout,
        )
        config.num_labels = 4
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Preprocess the data
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    logger.info("Tokenizing datasets")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_val_dataset = val_dataset.map(preprocess_function, batched=True)
    tokenized_holdout_test_dataset = test_dataset.map(preprocess_function, batched=True)

    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_val_dataset = tokenized_val_dataset.rename_column("label", "labels")
    tokenized_holdout_test_dataset = tokenized_holdout_test_dataset.rename_column("label", "labels")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=1000,
        save_steps=5000,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir='./logs',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16,
        greater_is_better=True,
        optim=args.optimizer
    )

    # Define a function to compute desired metrics
    def compute_metrics(p):
        accuracy_metric = evaluate.load("accuracy")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    class TrainEvalCallback(TrainerCallback):
        """
        Evaluate on the training set at epoch end and call the original compute_metrics
        which expects a (predictions, labels) tuple. Does NOT modify compute_metrics.
        Set eval_train_subset to an int to evaluate only first N train samples (fast).
        """
        def __init__(self, eval_train_subset: int | None = None):
            super().__init__()
            self.eval_train_subset = eval_train_subset
            self.trainer = None  # will set below

        def on_epoch_end(self, args, state, control, **kwargs):
            # Prefer an explicitly set trainer, fallback to kwargs
            trainer = self.trainer or kwargs.get("trainer")
            if trainer is None:
                logger.warning("TrainEvalCallback: no trainer available; skipping train eval.")
                return

            # Choose dataset (optionally subset for speed)
            ds = trainer.train_dataset
            if self.eval_train_subset is not None:
                try:
                    n = min(self.eval_train_subset, len(ds))
                    ds = ds.select(range(n))
                except Exception:
                    # if .select not available or fails, fall back to whole dataset
                    ds = trainer.train_dataset

            try:
                # Use trainer.predict to get logits and label_ids (does not call compute_metrics)
                pred_out = trainer.predict(ds)
                predictions = pred_out.predictions
                labels = pred_out.label_ids

                # If label_ids is None, extract labels from the dataset
                if labels is None:
                    # This works for HuggingFace datasets (column name "labels" or "label")
                    if "labels" in ds.column_names:
                        labels = np.array(ds["labels"])
                    elif "label" in ds.column_names:
                        labels = np.array(ds["label"])
                    else:
                        # fallback: try to build from examples
                        labels = np.array([ex.get("labels", ex.get("label")) for ex in ds])

                # Call your original compute_metrics which expects a tuple (predictions, labels)
                metrics = compute_metrics((predictions, labels))

                # Prefix with 'train_' so log_history and your plotting picks up entries
                metrics = {f"train_{k}": v for k, v in metrics.items()}

                # Log so Trainer stores it in state.log_history (and to TB/console)
                trainer.log(metrics)
                logger.info(f"[TrainEvalCallback] epoch={state.epoch:.2f} train metrics: {metrics}")

            except Exception as e:
                logger.warning(f"[TrainEvalCallback] failed to evaluate/train-metrics: {e}")

    # Initialize Trainer
    logger.info("Initializing Trainer")
    callbacks = []
    if args.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    train_eval_cb = TrainEvalCallback(eval_train_subset=None)  # or e.g. 2000
    train_eval_cb.trainer = trainer        # attach the trainer instance
    trainer.add_callback(train_eval_cb) 

    # Train the model
    logger.info("Starting training")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

    test_metrics = trainer.evaluate(eval_dataset=tokenized_holdout_test_dataset, metric_key_prefix="test")
    print(test_metrics)


    logs = trainer.state.log_history

    # train loss (from step logs) + train acc (from callback at epoch end)
    tr_epochs_loss, tr_losses = [], []
    tr_epochs_acc, tr_accs = [], []
    for e in logs:
        if "loss" in e and "epoch" in e and "learning_rate" in e:
            tr_epochs_loss.append(e["epoch"]); tr_losses.append(e["loss"])
        if "train_accuracy" in e and "epoch" in e:
            tr_epochs_acc.append(e["epoch"]); tr_accs.append(e["train_accuracy"])

    # val loss/acc (from built-in eval)
    va_epochs_loss, va_losses, va_epochs_acc, va_accs = [], [], [], []
    for e in logs:
        if "eval_loss" in e and "epoch" in e:
            va_epochs_loss.append(e["epoch"]); va_losses.append(e["eval_loss"])
        if "eval_accuracy" in e and "epoch" in e:
            va_epochs_acc.append(e["epoch"]); va_accs.append(e["eval_accuracy"])

    os.makedirs('./results', exist_ok=True)

    # Plot 1: Loss vs Epoch (different color/pattern)
    plt.figure(figsize=(7,5))
    plt.plot(tr_epochs_loss, tr_losses, label='Train Loss', linestyle='-', marker='o')
    plt.plot(va_epochs_loss, va_losses, label='Val Loss', linestyle='--', marker='s')
    plt.title('Loss vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig('./results/loss_vs_epochs.png', bbox_inches='tight', dpi=150); plt.close()

    # Plot 2: Accuracy vs Epoch (different color/pattern)
    plt.figure(figsize=(7,5))
    plt.plot(tr_epochs_acc, tr_accs, label='Train Acc', linestyle='-', marker='o')
    plt.plot(va_epochs_acc, va_accs, label='Val Acc', linestyle='--', marker='^')
    plt.title('Accuracy vs Epoch'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.ylim(0.8,1)
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig('./results/accuracy_vs_epochs.png', bbox_inches='tight', dpi=150); plt.close()

    # Save the trained model and tokenizer
    logger.info("Saving the model and tokenizer")
    model.save_pretrained('./results/final_model')
    tokenizer.save_pretrained('./results/final_model')

    logger.info("Script finished successfully")

if __name__ == "__main__":
    args = parse_args()
    main(args)



