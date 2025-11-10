import logging
import argparse
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, EarlyStoppingCallback, RobertaConfig
from datasets import load_dataset
import evaluate
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='Train RoBERTa on AG News with feature flags')
    parser.add_argument('--override-dropout', action='store_true', help='Override dropout via config')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout probability to set when overriding')
    parser.add_argument('--use-early-stopping', action='store_true', help='Enable early stopping callback')
    parser.add_argument('--early-stopping-patience', type=int, default=2, help='Patience for early stopping')
    parser.add_argument('--fp16', action='store_true', help='Enable mixed-precision training')
    parser.add_argument('--train-batch-size', type=int, default=8, help='Per-device train batch size')
    parser.add_argument('--eval-batch-size', type=int, default=8, help='Per-device eval batch size')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    return parser.parse_args()

def main(args):
    # Load AG News dataset
    logger.info("Loading AG News dataset")
    dataset = load_dataset("ag_news")
    train_dataset = dataset['train']
    test_dataset = dataset['test']

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
        # ensure the classification head has the right number of labels via the config
        config.num_labels = 4
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', config=config)
    else:
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=4)

    # Preprocess the data
    def preprocess_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

    logger.info("Tokenizing datasets")
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True)
    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True)
    tokenized_train_dataset = tokenized_train_dataset.rename_column("label", "labels")
    tokenized_test_dataset = tokenized_test_dataset.rename_column("label", "labels")

    # Define training arguments (respect CLI feature flags)
    training_args = TrainingArguments(
        output_dir='./results',
        eval_strategy='steps',
        save_strategy='steps',
        eval_steps=500,
        save_steps=5000,
        learning_rate=2e-5,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16
    )

    # Define a function to compute desired metrics
    def compute_metrics(p):
        accuracy_metric = evaluate.load("accuracy")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        return accuracy_metric.compute(predictions=predictions, references=labels)

    # Initialize Trainer
    logger.info("Initializing Trainer")
    callbacks = []
    if args.use_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train the model
    logger.info("Starting training")
    # Print CLI arg values for debugging / reproducibility
    try:
        logger.info("Parsed arguments:")
        for k, v in vars(args).items():
            logger.info(f"  {k}: {v}")
    except Exception:
        # fallback if args is not present or not a namespace
        logger.info(f"args: {args}")
        
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"An error occurred during training: {e}")
        raise

    # Evaluate the model
    logger.info("Evaluating the model")
    results = trainer.evaluate()
    logger.info(f"Evaluation results: {results}")

    # Save the trained model and tokenizer
    logger.info("Saving the model and tokenizer")
    model.save_pretrained('./results/final_model')
    tokenizer.save_pretrained('./results/final_model')

    logger.info("Script finished successfully")

if __name__ == "__main__":
    args = parse_args()
    main(args)
