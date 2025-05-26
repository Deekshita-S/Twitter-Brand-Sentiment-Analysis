from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(axis=-1)

    cls_report = classification_report(labels, preds, output_dict=True)

    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average='macro'),
        "weighted_f1": f1_score(labels, preds, average='weighted'),
        "macro_precision": precision_score(labels, preds, average='macro'),
        "macro_recall": recall_score(labels, preds, average='macro'),
        **{f"class_{int(k)}_f1": v["f1-score"] for k, v in cls_report.items() if k.isdigit()}
    }
