import wandb

def init_wandb():
    wandb.login(key="cb44241611614cf397e8c3cbe28c6f718de91dab")  # Or just use `wandb.login()`
    wandb.init(
        project="bert-sentiment-analysis",
        config={
            "batch_size": 16,
            "epochs": 10,
            "model": "bert-base-uncased",
            "learning_rate": 2e-5,
        }
    )
