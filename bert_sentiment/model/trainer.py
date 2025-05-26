
from torch import nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.model.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Add **kwargs to accept additional arguments. This allows the method to accept any additional arguments that the parent class might pass, including num_items_in_batch, while ignoring them since we don't need them for our custom loss calculation.
        labels = inputs.get("labels")
        # Remove labels from inputs if they exist
        inputs.pop("labels", None)
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
    
