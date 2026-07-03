"""
trainer.py
==========
Training, evaluation, prediction, feature extraction, and visualization
(PCA / UMAP / t-SNE / confidence analysis) utilities.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import f1_score

from config import device, THRESHOLD


# ------------------------------------------------------------------
# Loss function
# ------------------------------------------------------------------
def loss_fn(outputs, targets):
    """
    BCEWithLogitsLoss: combines a Sigmoid layer with BCELoss in a single,
    more numerically stable operation (via the log-sum-exp trick).
    """
    return nn.BCEWithLogitsLoss()(outputs, targets)


# ------------------------------------------------------------------
# Train / Eval loops
# ------------------------------------------------------------------
def train_model(training_loader, model, optimizer):
    """
    Train the model for one epoch. Returns the updated model, training
    accuracy, and mean loss.
    """
    losses = []
    correct_predictions = 0
    num_samples = 0
    model.train()

    loop = tqdm(enumerate(training_loader), total=len(training_loader), leave=True, colour="steelblue")

    for batch_idx, data in loop:
        ids = data["input_ids"].to(device, dtype=torch.long)
        mask = data["attention_mask"].to(device, dtype=torch.long)
        targets = data["targets"].to(device, dtype=torch.float)

        outputs = model(ids, mask)
        loss = loss_fn(outputs, targets)
        losses.append(loss.item())

        outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
        targets_np = targets.cpu().detach().numpy()
        correct_predictions += np.sum(outputs == targets_np)
        num_samples += targets_np.size

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return model, float(correct_predictions) / num_samples, np.mean(losses)


def eval_model(validation_loader, model, optimizer):
    """
    Evaluate the model on a validation/test set. Returns accuracy and mean loss.
    """
    losses = []
    correct_predictions = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_loader, desc="Evaluating"), 0):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            outputs = model(ids, mask)

            loss = loss_fn(outputs, targets)
            losses.append(loss.item())

            outputs = torch.sigmoid(outputs).cpu().detach().numpy().round()
            targets_np = targets.cpu().detach().numpy()
            correct_predictions += np.sum(outputs == targets_np)
            num_samples += targets_np.size

    return float(correct_predictions) / num_samples, np.mean(losses)


def get_predictions(model, df, data_loader, epoch, data_dir):
    """
    Run inference on a dataset, log incorrect predictions and confidence
    scores, then return everything sorted back into the original index order.
    """
    model = model.eval()

    titles, predictions, prediction_probs = [], [], []
    target_values, confidence_scores, indices = [], [], []
    incorrect_predictions = []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Predicting"):
            title = data["title"]
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)
            index = data["index"].to(device, dtype=torch.long)

            outputs = model(ids, mask)
            logits = outputs

            outputs = torch.sigmoid(outputs).detach().cpu()
            preds = (outputs >= THRESHOLD).float()
            targets = targets.detach().cpu()

            titles.extend(title)
            predictions.extend(preds)
            prediction_probs.extend(outputs)
            target_values.extend(targets)
            indices.extend(index.cpu().numpy())

            probs = torch.softmax(logits, dim=1)
            confidence = probs.max(dim=1).values.cpu().numpy()
            confidence_scores.extend(confidence)

            first_label_index = 0
            incorrect_indices = (
                preds[:, first_label_index] != targets[:, first_label_index]
            ).nonzero(as_tuple=True)[0]
            for idx in incorrect_indices:
                incorrect_predictions.append({
                    "title": title[idx],
                    "true_label": targets[idx, first_label_index].item(),
                    "predicted_label": preds[idx, first_label_index].item(),
                    "confidence": confidence[idx],
                })

    indices = np.array(indices)
    sorted_indices = np.argsort(indices)

    titles = np.array(titles)[sorted_indices]
    predictions = torch.stack(predictions)[sorted_indices]
    prediction_probs = torch.stack(prediction_probs)[sorted_indices]
    target_values = torch.stack(target_values)[sorted_indices]
    confidence_scores = np.array(confidence_scores)[sorted_indices]

    df_confidence = df.copy()
    df_confidence["confidence"] = confidence_scores

    incorrect_predictions_df = pd.DataFrame(incorrect_predictions)
    print("incorrect_predictions_df:", len(incorrect_predictions_df))

    positive_samples = incorrect_predictions_df[incorrect_predictions_df["true_label"] == 1]
    negative_samples = incorrect_predictions_df[incorrect_predictions_df["true_label"] == 0]
    print(f"Epoch {epoch} - Positive Samples: {len(positive_samples)}")
    print(f"Epoch {epoch} - Negative Samples: {len(negative_samples)}")

    incorrect_predictions_df.to_csv(
        os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch}.csv"), index=False
    )
    df_confidence.to_csv(
        os.path.join(data_dir, f"df_train_adjusted_confidence_epoch_{epoch}.csv"), index=False
    )

    return titles, predictions, prediction_probs, target_values, df_confidence, incorrect_predictions_df


def get_confidence_scores(model, df, tokenizer, max_len, target_list):
    """
    Build a dataset/dataloader on the fly, compute per-sample confidence
    scores, and merge them back into the original DataFrame.
    """
    from dataset import CustomDataset  # local import to avoid a circular import

    dataset = CustomDataset(df, tokenizer, max_len, target_list)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    model.eval()
    confidence_scores = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids=input_ids, attn_mask=attention_mask)
            probs = torch.softmax(outputs, dim=1)
            confidence = probs.max(dim=1).values.cpu().numpy()
            confidence_scores.extend(confidence)

    df_confidence = df.copy()
    df_confidence["confidence"] = confidence_scores
    return df_confidence


def extract_features(model, data_loader, epoch, data_dir):
    """
    Extract the model's pre-classification outputs (logits) as features,
    for use in PCA/UMAP/t-SNE visualization.
    """
    model.eval()
    features_list, labels_list = [], []

    with torch.no_grad():
        for data in tqdm(data_loader, desc="Extracting Features"):
            ids = data["input_ids"].to(device, dtype=torch.long)
            mask = data["attention_mask"].to(device, dtype=torch.long)
            targets = data["targets"].to(device, dtype=torch.float)

            outputs = model(ids, mask)
            features_list.append(outputs.cpu().numpy())
            labels_list.append(targets.cpu().numpy())

    features = np.vstack(features_list)
    labels = np.vstack(labels_list)
    return features, labels


def calculate_f1_per_label(y_true, y_pred):
    """Compute the F1 score for each label."""
    f1_scores = []
    for i in range(y_true.shape[1]):
        f1_scores.append(f1_score(y_true[:, i], y_pred[:, i]))
    return f1_scores


def calculate_confidence_intervals(prediction_probs, target_list, target_values, epoch, data_dir="."):
    """
    Compute the confidence-score distribution (positive/negative histograms)
    for each label and save the results to a CSV file.
    """
    confidence_ranges = []
    results = []

    for i in range(prediction_probs.shape[1]):
        label_probs = prediction_probs[:, i]
        label_targets = target_values[:, i]

        positive_probs = label_probs[label_targets == 1]
        negative_probs = label_probs[label_targets == 0]

        confidence_range = (label_probs.min(), label_probs.max())
        confidence_ranges.append(confidence_range)

        bins = np.arange(0, 1.1, 0.1)
        positive_hist, _ = np.histogram(positive_probs, bins=bins)
        negative_hist, _ = np.histogram(negative_probs, bins=bins)

        print(f"Label {target_list[i]} confidence range: {confidence_range}, count: {len(label_probs)}")

        results.append({
            "label": target_list[i],
            "confidence_range_min": confidence_range[0],
            "confidence_range_max": confidence_range[1],
            "positive_hist": positive_hist.tolist(),
            "negative_hist": negative_hist.tolist(),
        })

    results_df = pd.DataFrame(results)
    results_path = os.path.join(data_dir, f"confidence_intervals_epoch_{epoch}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Confidence intervals saved to {results_path}")

    return confidence_ranges


# ------------------------------------------------------------------
# Visualization: PCA / UMAP / t-SNE
# ------------------------------------------------------------------
CUSTOM_LABELS = [
    "Cytoplasm", "Nucleus", "Extracellular", "Cell membrane",
    "Mitochondrion", "Plastid", "Endoplasmic reticulum",
    "Lysosome/Vacuole", "Golgi apparatus", "Peroxisome",
]


def _save_scatter(result, labels, title, epoch, data_dir):
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(result[:, 0], result[:, 1], c=labels.argmax(axis=1), cmap="viridis", s=5)
    plt.title(title)

    cbar = plt.colorbar(scatter)
    cbar.set_ticks(range(10))
    cbar.set_ticklabels(CUSTOM_LABELS)
    cbar.set_label("Protein Localization Categories")

    plt.show()
    filename = f"{title}_epoch_{epoch}.png"
    plt.savefig(os.path.join(data_dir, filename))
    plt.close()


def plot_pca(features, labels, title, epoch, data_dir):
    pca_result = PCA(n_components=2).fit_transform(features)
    _save_scatter(pca_result, labels, title, epoch, data_dir)


def plot_umap(features, labels, title, epoch, data_dir):
    umap_result = umap.UMAP().fit_transform(features)
    _save_scatter(umap_result, labels, title, epoch, data_dir)


def plot_tsne(features, labels, title, epoch, data_dir):
    tsne_result = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(features)
    _save_scatter(tsne_result, labels, title, epoch, data_dir)


def _save_binary_scatter(result, labels, title, epoch, data_dir):
    zero_count = np.sum(labels[:, 0] == 0)
    one_count = np.sum(labels[:, 0] == 1)

    plt.figure(figsize=(10, 8))
    plt.scatter(result[labels[:, 0] == 0, 0], result[labels[:, 0] == 0, 1],
                c="blue", label=f"Cytoplasm 0 ({zero_count})", s=5)
    plt.scatter(result[labels[:, 0] == 1, 0], result[labels[:, 0] == 1, 1],
                c="red", label=f"Cytoplasm 1 ({one_count})", s=5)
    plt.title(title)
    plt.legend()
    plt.show()

    filename = f"{title}_epoch_{epoch}.png"
    plt.savefig(os.path.join(data_dir, filename))
    plt.close()


def plot_pca_confidence(features, labels, title, epoch, data_dir):
    pca_result = PCA(n_components=2).fit_transform(features)
    _save_binary_scatter(pca_result, labels, title, epoch, data_dir)


def plot_umap_confidence(features, labels, title, epoch, data_dir):
    umap_result = umap.UMAP().fit_transform(features)
    _save_binary_scatter(umap_result, labels, title, epoch, data_dir)


def plot_tsne_confidence(features, labels, title, epoch, data_dir):
    tsne_result = TSNE(n_components=2, perplexity=30, learning_rate=200).fit_transform(features)
    _save_binary_scatter(tsne_result, labels, title, epoch, data_dir)


def plot_confidence_distribution(df_confidence, epoch, save_path=None):
    """
    Plot the distribution of sample confidence scores, overlaid with a
    fitted normal distribution curve for comparison.
    """
    bins = np.arange(0, 1.1, 0.1)
    confidence_counts, _ = np.histogram(df_confidence["confidence"], bins=bins, density=False)
    print("Sample count per bin:", confidence_counts)

    bin_mids = (bins[:-1] + bins[1:]) / 2

    mean = df_confidence["confidence"].mean()
    std = df_confidence["confidence"].std()
    x = np.linspace(0, 1, 100)
    y = norm.pdf(x, mean, std) * len(df_confidence) * np.diff(bins).mean()

    plt.figure(figsize=(10, 6))
    plt.bar(bin_mids, confidence_counts, width=0.1, color="b", alpha=0.5, label="Confidence Distribution")
    plt.plot(x, y, color="r", label="Normal Distribution")
    plt.title(f"Confidence Distribution vs Normal Distribution - Epoch {epoch}")
    plt.xlabel("Confidence Score")
    plt.ylabel("Number of Samples")
    plt.xticks(bins)
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
