"""
main.py
=======
Main pipeline: load data -> build model -> training loop (with checkpoint
resume support) -> repeated evaluation (Accuracy / F1 / MCC / ensemble).

Usage:
    python main.py
"""

import os
import gc
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import T5Tokenizer, AdamW
from sklearn.metrics import matthews_corrcoef, classification_report

from config import (
    device, DATA_DIR, PRETRAINED_MODEL_NAME, MAX_LEN,
    LEARNING_RATE, EPOCHS, TEST_BATCH_SIZE, LABEL_COLUMNS,
    CHECKPOINT_PATH, BEST_MODEL_PATH, NUM_EVAL_RUNS,
)
from dataset import (
    load_raw_data, split_train_test, get_label_counts,
    get_target_list, build_dataloaders, CustomDataset,
)
from model import build_model, T5Class
from trainer import (
    train_model, eval_model, get_predictions, get_confidence_scores,
    extract_features, calculate_f1_per_label, calculate_confidence_intervals,
    plot_pca, plot_umap, plot_tsne, plot_confidence_distribution,
)


def compare_epoch_predictions(epoch1, epoch2, data_dir):
    """
    Compare incorrect predictions between two epochs, split samples into
    consistent vs. inconsistent predictions, and save the results.
    """
    epoch1_path = os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch1}.csv")
    epoch2_path = os.path.join(data_dir, f"incorrect_predictions_epoch_{epoch2}.csv")

    df_epoch1 = pd.read_csv(epoch1_path)
    df_epoch2 = pd.read_csv(epoch2_path)

    same_predictions, different_predictions = [], []
    for index, row in df_epoch1.iterrows():
        title = row["title"]
        true_label = row["true_label"]
        predicted_label_epoch1 = row["predicted_label"]
        confidence_epoch1 = row["confidence"]

        matching_row = df_epoch2[df_epoch2["title"] == title]
        if not matching_row.empty:
            predicted_label_epoch2 = matching_row["predicted_label"].values[0]
            confidence_epoch2 = matching_row["confidence"].values[0]

            record = {
                "title": title,
                "true_label": true_label,
                "predicted_label_epoch1": predicted_label_epoch1,
                "predicted_label_epoch2": predicted_label_epoch2,
                "confidence_epoch1": confidence_epoch1,
                "confidence_epoch2": confidence_epoch2,
            }
            if predicted_label_epoch1 == predicted_label_epoch2:
                same_predictions.append(record)
            else:
                different_predictions.append(record)

    same_df = pd.DataFrame(same_predictions)
    diff_df = pd.DataFrame(different_predictions)

    same_df.to_csv(os.path.join(data_dir, f"same_predictions_epoch_{epoch1}_vs_{epoch2}.csv"), index=False)
    diff_df.to_csv(os.path.join(data_dir, f"different_predictions_epoch_{epoch1}_vs_{epoch2}.csv"), index=False)

    print(f"changed_samples_df: {len(same_df)}")
    if not same_df.empty:
        positive_samples = same_df[same_df["true_label"] == 1]
        negative_samples = same_df[same_df["true_label"] == 0]
        print(f"Changed Positive Samples: {len(positive_samples)}")
        print(f"Changed Negative Samples: {len(negative_samples)}")
    else:
        print("No changed samples found.")

    return same_df, diff_df


def run_training(model, optimizer, tokenizer, df_train, df_test,
                  train_data_loader, test_data_loader, target_list,
                  data_dir=DATA_DIR, epochs=EPOCHS):
    """
    Main training loop: train -> validate -> predict -> extract features
    -> visualize -> save best model/checkpoint. Supports resuming from a
    saved checkpoint.
    """
    history = defaultdict(list)
    best_accuracy = 0
    start_epoch = 1

    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        best_accuracy = checkpoint["best_accuracy"]
        history = checkpoint["history"]
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded, resuming training from epoch {start_epoch}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        torch.cuda.empty_cache()
        print("CUDA memory cache cleared")

        model, train_acc, train_loss = train_model(train_data_loader, model, optimizer)
        test_acc, test_loss = eval_model(test_data_loader, model, optimizer)

        (titles, predictions, prediction_probs, target_values,
         df_confidence, incorrect_predictions_df) = get_predictions(
            model, df_test, test_data_loader, epoch, data_dir
        )

        train_features, train_labels = extract_features(model, train_data_loader, epoch, data_dir)
        test_features, test_labels = extract_features(model, test_data_loader, epoch, data_dir)

        avg_confidence = df_confidence["confidence"].mean()
        print(f"Adjusted training set average confidence: {avg_confidence:.4f}")
        df_confidence.to_csv(os.path.join(data_dir, f"df_train_adjusted_confidence_epoch_{epoch}.csv"), index=False)

        if epoch > 1:
            compare_epoch_predictions(epoch - 1, epoch, data_dir)

        plot_pca(train_features, train_labels, f"PCA of Training Set Features_Epoch:{epoch}", epoch, data_dir)
        plot_pca(test_features, test_labels, f"PCA of Test Set Features_Epoch:{epoch}", epoch, data_dir)
        plot_umap(train_features, train_labels, f"UMAP of Training Set Features_Epoch:{epoch}", epoch, data_dir)
        plot_umap(test_features, test_labels, f"UMAP of Test Set Features_Epoch:{epoch}", epoch, data_dir)
        plot_tsne(train_features, train_labels, f"t-SNE of Training Set Features_Epoch:{epoch}", epoch, data_dir)
        plot_tsne(test_features, test_labels, f"t-SNE of Test Set Features_Epoch:{epoch}", epoch, data_dir)

        df_train_confidence = get_confidence_scores(model, df_train, tokenizer, MAX_LEN, target_list)
        avg_train_confidence = df_train_confidence["confidence"].mean()
        print(f"Training set average confidence: {avg_train_confidence:.4f}")

        plot_confidence_distribution(
            df_train_confidence, epoch,
            save_path=os.path.join(data_dir, f"confidence_distribution_epoch_{epoch}.png"),
        )
        df_train_confidence.to_csv(os.path.join(data_dir, f"df_train_confidence_epoch_{epoch}.csv"), index=False)

        print(f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f} "
              f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["test_acc"].append(test_acc)
        history["test_loss"].append(test_loss)

        confidence_ranges = calculate_confidence_intervals(
            prediction_probs, target_list, target_values, epoch, data_dir
        )

        f1_scores = calculate_f1_per_label(target_values, predictions)
        print(f"F1 scores per label: {f1_scores}")

        if test_acc > best_accuracy:
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            best_accuracy = test_acc
            print(f"Best model saved with test_acc={test_acc:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "history": history,
        }, CHECKPOINT_PATH)
        print(f"Checkpoint saved for epoch {epoch}")

        with open(os.path.join(data_dir, "training_log.txt"), "a") as log_file:
            log_file.write(f"Epoch {epoch}/{epochs}\n")
            log_file.write(f"train_loss={train_loss:.4f}, test_loss={test_loss:.4f} "
                            f"train_acc={train_acc:.4f}, test_acc={test_acc:.4f}\n")
            log_file.write(f"Confidence ranges: {confidence_ranges}\n")
            log_file.write(f"Adjusted training set average confidence: {avg_confidence:.4f}\n")
            log_file.write(f"Training set average confidence: {avg_train_confidence:.4f}\n")
            log_file.write(f"F1 scores per label: {f1_scores}\n")
            log_file.write(f"Best accuracy: {best_accuracy:.4f}\n")
            log_file.write("-" * 50 + "\n")

    print("Training completed.")
    return model, history


def run_evaluation(tokenizer, df_test, target_list, data_dir=DATA_DIR,
                    num_runs=NUM_EVAL_RUNS):
    """
    Load the best saved model and evaluate it num_runs times, computing
    Accuracy / F1 / MCC each time, then ensemble the predictions (mean
    aggregation) for a final evaluation.
    """
    all_test_accs, all_macro_f1s, all_micro_f1s = [], [], []
    all_mcc_per_label, all_results = [], []
    all_predictions, all_targets = [], []

    for run in range(num_runs):
        torch.cuda.empty_cache()

        model = T5Class()
        model.load_state_dict(torch.load(BEST_MODEL_PATH))
        model = model.to(device)
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        test_dataset = CustomDataset(df_test, tokenizer, MAX_LEN, target_list)
        test_data_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True, num_workers=0
        )
        test_acc, test_loss = eval_model(test_data_loader, model, optimizer)

        (titles, predictions, prediction_probs, target_values,
         df_confidence, incorrect_predictions_df) = get_predictions(
            model, df_test, test_data_loader, f"final_prediction_{run + 1}", data_dir
        )

        all_predictions.append(predictions)
        all_targets.append(target_values)

        n_classes = target_values.shape[1]
        mcc_per_label = [
            matthews_corrcoef(target_values[:, i], predictions[:, i]) for i in range(n_classes)
        ]

        report = classification_report(
            target_values, predictions, target_names=target_list, digits=4, output_dict=True
        )

        all_test_accs.append(test_acc)
        all_macro_f1s.append(report["macro avg"]["f1-score"])
        all_micro_f1s.append(report["micro avg"]["f1-score"])
        all_mcc_per_label.append(mcc_per_label)

        run_result = {
            "Run": run + 1,
            "Test Accuracy": test_acc,
            "Macro F1": report["macro avg"]["f1-score"],
            "Micro F1": report["micro avg"]["f1-score"],
            "MCC per Label": mcc_per_label,
        }
        if any(str(i) in report for i in range(n_classes)):
            for i, label in enumerate(target_list):
                run_result[f"{label} F1"] = report[str(i)]["f1-score"]
                run_result[f"{label} Precision"] = report[str(i)]["precision"]
                run_result[f"{label} Recall"] = report[str(i)]["recall"]
        else:
            print("Warning: classification report does not contain per-label metrics.")
            for i, label in enumerate(target_list):
                run_result[f"{label} F1"] = report["macro avg"]["f1-score"]
                run_result[f"{label} Precision"] = report["macro avg"]["precision"]
                run_result[f"{label} Recall"] = report["macro avg"]["recall"]

        all_results.append(run_result)

        print(f"Run {run + 1} - Test Accuracy: {test_acc:.4f}")
        print(f"Run {run + 1} - Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"Run {run + 1} - Micro F1: {report['micro avg']['f1-score']:.4f}")
        print(f"Run {run + 1} - MCC per label: {mcc_per_label}")

        gc.collect()
        torch.cuda.empty_cache()

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(data_dir, "evaluation_results.csv"), index=False)
    print("Evaluation results saved to evaluation_results.csv")

    print("Overall Results:")
    print(f"Average Test Accuracy: {np.mean(all_test_accs):.4f} +/- {np.std(all_test_accs):.4f}")
    print(f"Average Macro F1: {np.mean(all_macro_f1s):.4f} +/- {np.std(all_macro_f1s):.4f}")
    print(f"Average Micro F1: {np.mean(all_micro_f1s):.4f} +/- {np.std(all_micro_f1s):.4f}")
    print(f"Average MCC per label: {np.mean(all_mcc_per_label, axis=0)} +/- {np.std(all_mcc_per_label, axis=0)}")

    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    assert all_predictions.shape[1:] == all_targets.shape[1:], "Shape mismatch between predictions and targets"

    ensemble_predictions = np.mean(all_predictions, axis=0)
    ensemble_predictions = (ensemble_predictions > 0.5).astype(int)

    ensemble_report = classification_report(
        all_targets[0], ensemble_predictions, target_names=target_list, digits=4, output_dict=True
    )
    if "accuracy" not in ensemble_report:
        ensemble_accuracy = np.mean((ensemble_predictions == all_targets[0]).all(axis=1))
        ensemble_report["accuracy"] = ensemble_accuracy

    n_classes = all_targets[0].shape[1]
    ensemble_mcc_per_label = [
        matthews_corrcoef(all_targets[0][:, i], ensemble_predictions[:, i]) for i in range(n_classes)
    ]

    print("Ensemble Results:")
    print(f"Ensemble Test Accuracy: {ensemble_report['accuracy']:.4f}")
    print(f"Ensemble Macro F1: {ensemble_report['macro avg']['f1-score']:.4f}")
    print(f"Ensemble Micro F1: {ensemble_report['micro avg']['f1-score']:.4f}")
    print(f"Ensemble MCC per label: {ensemble_mcc_per_label}")

    ensemble_results = {
        "Ensemble Test Accuracy": ensemble_report["accuracy"],
        "Ensemble Macro F1": ensemble_report["macro avg"]["f1-score"],
        "Ensemble Micro F1": ensemble_report["micro avg"]["f1-score"],
        "Ensemble MCC per Label": ensemble_mcc_per_label,
    }
    for i, label in enumerate(target_list):
        ensemble_results[f"{label} F1"] = ensemble_report[str(i)]["f1-score"]
        ensemble_results[f"{label} Precision"] = ensemble_report[str(i)]["precision"]
        ensemble_results[f"{label} Recall"] = ensemble_report[str(i)]["recall"]

    ensemble_results_df = pd.DataFrame([ensemble_results])
    ensemble_results_df.to_csv(os.path.join(data_dir, "ensemble_evaluation_results.csv"), index=False)
    print("Ensemble evaluation results saved to ensemble_evaluation_results.csv")

    return results_df, ensemble_results_df


def main():
    # 1. Load tokenizer and data
    tokenizer = T5Tokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
    df_data = load_raw_data()
    df_train, df_test = split_train_test(df_data)

    print("Train dataset label counts:")
    print(get_label_counts(df_train))
    print("\nTest dataset label counts:")
    print(get_label_counts(df_test))

    target_list = get_target_list(df_data)

    # 2. Build datasets/dataloaders
    train_dataset, test_dataset, train_data_loader, test_data_loader = build_dataloaders(
        df_train, df_test, tokenizer, target_list
    )

    # 3. Build model and optimizer
    model = build_model(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # 4. Train
    model, history = run_training(
        model, optimizer, tokenizer, df_train, df_test,
        train_data_loader, test_data_loader, target_list,
    )

    # 5. Repeated evaluation and ensembling
    run_evaluation(tokenizer, df_test, target_list)


if __name__ == "__main__":
    main()
