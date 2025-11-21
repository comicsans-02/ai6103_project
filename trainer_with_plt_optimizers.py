import logging
import argparse
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, DistilBertConfig, TrainerCallback
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
    # Updated defaults for no regularization baseline
    parser = argparse.ArgumentParser(description='Train DistilBERT on AG News with multiple optimizers')
    parser.add_argument('--override-dropout', action='store_true', default=True, help='Override dropout via config')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout probability to set when overriding')
    parser.add_argument('--use-early-stopping', action='store_true', help='Enable early stopping callback')
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--train-batch-size', type=int, default=16, help='Per-device train batch size')
    parser.add_argument('--eval-batch-size', type=int, default=16, help='Per-device eval batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay')
    return parser.parse_args()

def train_with_optimizer(optimizer_name, args, train_dataset, val_dataset, test_dataset, tokenizer):
    logger.info(f"\n{'='*50}\nTraining with optimizer: {optimizer_name}\n{'='*50}")
    
    # Load the model
    if args.override_dropout:
        logger.info(f"Overriding dropout in config to {args.dropout}")
        config = DistilBertConfig(
            dropout=args.dropout,
            attention_dropout=args.dropout,
            qa_dropout=args.dropout,
            seq_classif_dropout=args.dropout,
            vocab_size=tokenizer.vocab_size,
        )
        config.num_labels = 4
        model = DistilBertForSequenceClassification(config=config)
    else:
        config = DistilBertConfig(vocab_size=tokenizer.vocab_size)
        config.num_labels = 4
        model = DistilBertForSequenceClassification(config=config)
    
    # Define training arguments
    output_dir = f'./results_{optimizer_name}'
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=1000,
        save_steps=5000,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=f'./logs_{optimizer_name}',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16,
        greater_is_better=True,
        optim=optimizer_name,
        report_to="none"
    )
    
    # Define a function to compute desired metrics
    def compute_metrics(p):
        accuracy_metric = evaluate.load("accuracy")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)
    
    class TrainEvalCallback(TrainerCallback):
      
        def __init__(self, eval_train_subset=None):
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
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
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
    
    valuation_metrics = trainer.evaluate(eval_dataset=val_dataset, metric_key_prefix="valuation")
    print(f"{optimizer_name} valuation metrics:", valuation_metrics)
    
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
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Loss vs Epoch (different color/pattern)
    plt.figure(figsize=(7,5))
    plt.plot(tr_epochs_loss, tr_losses, label='Train Loss', linestyle='-', marker='o')
    plt.plot(va_epochs_loss, va_losses, label='Val Loss', linestyle='--', marker='s')
    plt.title(f'Loss vs Epoch ({optimizer_name})'); plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(f'{output_dir}/loss_vs_epochs_{optimizer_name}.png', bbox_inches='tight', dpi=150); plt.close()
    
    # Plot 2: Accuracy vs Epoch (different color/pattern) - Dynamic y-axis
    plt.figure(figsize=(7,5))
    plt.plot(tr_epochs_acc, tr_accs, label='Train Acc', linestyle='-', marker='o')
    plt.plot(va_epochs_acc, va_accs, label='Val Acc', linestyle='--', marker='^')
    plt.title(f'Accuracy vs Epoch ({optimizer_name})'); plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    # Dynamic y-axis limits to avoid cut-off (especially for SGD)
    if va_accs and tr_accs:
        min_acc = min(min(tr_accs), min(va_accs))
        max_acc = max(max(tr_accs), max(va_accs))
        margin = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 0.05
        plt.ylim(max(0, min_acc - margin), min(1.0, max_acc + margin))
    plt.grid(True, alpha=0.3); plt.legend()
    plt.savefig(f'{output_dir}/accuracy_vs_epochs_{optimizer_name}.png', bbox_inches='tight', dpi=150); plt.close()
    
    # Save the trained model and tokenizer
    logger.info("Saving the model and tokenizer")
    model.save_pretrained(f'{output_dir}/final_model')
    tokenizer.save_pretrained(f'{output_dir}/final_model')
    
    return {
        'optimizer': optimizer_name,
        'tr_epochs_loss': tr_epochs_loss, 'tr_losses': tr_losses,
        'tr_epochs_acc': tr_epochs_acc, 'tr_accs': tr_accs,
        'va_epochs_loss': va_epochs_loss, 'va_losses': va_losses,
        'va_epochs_acc': va_epochs_acc, 'va_accs': va_accs,
        'valuation_metrics': valuation_metrics
    }

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
    logger.info("Loading tokenizer")
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    
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
    
    # List of optimizers to try: AdamW, SGD, adagrad
    optimizers = ['adamw_torch', 'sgd', 'adagrad']
    
    # Train with each optimizer and collect results
    all_results = []
    for optimizer_name in optimizers:
        result = train_with_optimizer(
            optimizer_name, args,
            tokenized_train_dataset, tokenized_val_dataset, tokenized_holdout_test_dataset,
            tokenizer
        )
        all_results.append(result)
    
    # Create combined plots
    os.makedirs('./results_combined', exist_ok=True)
    
    # Combined Loss Plot
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']
    
    for i, result in enumerate(all_results):
        opt = result['optimizer']
        plt.plot(result['va_epochs_loss'], result['va_losses'],
                label=f'{opt}', linestyle='--', marker=markers[i],
                color=colors[i], markersize=6, linewidth=2)
    
    plt.title('Validation Loss vs Epoch (All Optimizers)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('./results_combined/combined_loss_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Combined Accuracy Plot - Dynamic y-axis
    plt.figure(figsize=(10, 6))
    
    for i, result in enumerate(all_results):
        opt = result['optimizer']
        plt.plot(result['va_epochs_acc'], result['va_accs'],
                label=f'{opt}', linestyle='--', marker=markers[i],
                color=colors[i], markersize=6, linewidth=2)
    
    plt.title('Validation Accuracy vs Epoch (All Optimizers)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    # Dynamic y-axis for combined plot
    all_accs = [acc for result in all_results for acc in result['va_accs']]
    if all_accs:
        min_acc = min(all_accs)
        max_acc = max(all_accs)
        margin = (max_acc - min_acc) * 0.1 if max_acc > min_acc else 0.05
        plt.ylim(max(0, min_acc - margin), min(1.0, max_acc + margin))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.savefig('./results_combined/combined_accuracy_comparison.png', bbox_inches='tight', dpi=150)
    plt.close()
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("TRAINING SUMMARY - ALL OPTIMIZERS")
    logger.info("="*80)
    logger.info(f"Configuration: Dropout={args.dropout}, Weight Decay={args.weight_decay}\n")
    
    # Create results table
    print("\n" + "="*80)
    print(f"{'Optimizer':<15} | {'Train Accuracy':>15} | {'Val Accuracy':>15} | {'Train Loss':>12} | {'Val Loss':>12}")
    print("-"*80)
    
    for result in all_results:
        opt = result['optimizer']
        train_acc = result['tr_accs'][-1] if result['tr_accs'] else 0.0
        val_acc = result['va_accs'][-1] if result['va_accs'] else 0.0
        train_loss = result['tr_losses'][-1] if result['tr_losses'] else 0.0
        val_loss = result['va_losses'][-1] if result['va_losses'] else 0.0
        
        print(f"{opt:<15} | {train_acc:>15.4f} | {val_acc:>15.4f} | {train_loss:>12.4f} | {val_loss:>12.4f}")
    
    print("="*80 + "\n")
    
    # Also save to file
    with open('./results_combined/summary_table.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Configuration: Dropout={args.dropout}, Weight Decay={args.weight_decay}\n")
        f.write("="*80 + "\n\n")
        f.write(f"{'Optimizer':<15} | {'Train Accuracy':>15} | {'Val Accuracy':>15} | {'Train Loss':>12} | {'Val Loss':>12}\n")
        f.write("-"*80 + "\n")
        
        for result in all_results:
            opt = result['optimizer']
            train_acc = result['tr_accs'][-1] if result['tr_accs'] else 0.0
            val_acc = result['va_accs'][-1] if result['va_accs'] else 0.0
            train_loss = result['tr_losses'][-1] if result['tr_losses'] else 0.0
            val_loss = result['va_losses'][-1] if result['va_losses'] else 0.0
            
            f.write(f"{opt:<15} | {train_acc:>15.4f} | {val_acc:>15.4f} | {train_loss:>12.4f} | {val_loss:>12.4f}\n")
        
        f.write("="*80 + "\n")
    
    logger.info("Script finished successfully")
    logger.info(f"Individual plots saved in: ./results_<optimizer_name>/")
    logger.info(f"Combined plots saved in: ./results_combined/")
    logger.info(f"Summary table saved in: ./results_combined/summary_table.txt")

if __name__ == "__main__":
    args = parse_args()
    main(args)