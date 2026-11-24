"""Accuracy, macro-F1 и матрица ошибок. Одна функция для val и test."""

import torch


def confusion_update(matrix, target, predicted):
    for true_label, pred_label in zip(target.tolist(), predicted.tolist()):
        matrix[true_label, pred_label] += 1


def scores_from_confusion(matrix):
    total = int(matrix.sum().item())
    accuracy = float(matrix.diag().sum().item()) / total if total else 0.0
    f1_values = []
    for class_index in range(matrix.shape[0]):
        tp = float(matrix[class_index, class_index].item())
        fp = float(matrix[:, class_index].sum().item()) - tp
        fn = float(matrix[class_index, :].sum().item()) - tp
        if tp == 0 and fp == 0 and fn == 0:
            f1_values.append(0.0)
            continue
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1_values.append(
            0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        )
    macro_f1 = sum(f1_values) / len(f1_values)
    return accuracy, macro_f1


def empty_confusion(num_classes):
    return torch.zeros(num_classes, num_classes, dtype=torch.int64)
