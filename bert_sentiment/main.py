import torch
import numpy as np
from datasets import Dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
from sklearn.metrics import classification_report
from data.preprocessing import load_and_augment
from model.trainer import WeightedTrainer
from utils.metrics import compute_metrics
from config.wandb_config import init_wandb


print("Starting data preprocessing...")
df = load_and_augment("Dataset - Train.csv")
print("Completed data preprocessing...")

dataset = Dataset.from_pandas(df[['text', 'sentiment']])
dataset = dataset.class_encode_column("sentiment")
dataset = dataset.train_test_split(test_size=0.2)

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(dataset['train']['sentiment']),
    y=dataset['train']['sentiment']
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    hidden_dropout_prob=0.3, 
    num_labels=3,
)

def tokenize_function(examples):
    tokens = tokenizer(examples["text"], padding="max_length", truncation=True)
    tokens["labels"] = examples["sentiment"]
    return tokens

tokenized = dataset.map(tokenize_function, batched=True)

args = TrainingArguments(
    output_dir="bert-output",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
    load_best_model_at_end=True,
    metric_for_best_model="eval_macro_f1",
    greater_is_better=True
)

init_wandb()

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=tokenized['train'],
    eval_dataset=tokenized['test'],
    compute_metrics=compute_metrics
)


print("Starting model training...")
trainer.train()
print("Completed model training...")


print("Saving model..")
model.save_pretrained("bert_model")
tokenizer.save_pretrained("bert_model")


# ------ Testing --------
print("Testing model...")
# Get predictions on test set
predictions = trainer.predict(tokenized['test'])
preds = predictions.predictions.argmax(axis=-1)

# Full classification report
print("\nDetailed Classification Report:")
print(classification_report(
    tokenized['test']['sentiment'], 
    preds,
    target_names=[f"Class {i}" for i in range(3)]
))
