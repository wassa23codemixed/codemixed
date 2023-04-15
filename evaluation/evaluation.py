#!/usr/bin/env python
# Author: roman.klinger@ims.uni-stuttgart.de
# Evaluation script for Empathy shared task at WASSA 2022
# Adapted for CodaLab purposes by Orphee (orphee.declercq@ugent.be) in May 2018
# Adapted for multiple subtasks by Valentin Barriere in December 2022 (python 3)

from __future__ import print_function
import sys
import argparse
import os
import pandas as pd
from math import sqrt

to_round = 4


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def read_mcec(filename):
    df = pd.read_csv(filename)
    return df['Emotion']


def read_mlec(filename):
    df = pd.read_csv(filename)
    df['target_list'] = df[
        ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'love', 'optimism', 'pessimism', 'sadness', 'surprise',
         'trust']].values.tolist()
    return df['target_list']


def jaccard_similarity(y_true, y_pred):
    # calculate the intersection and union of the label sets for each sample
    intersection = [set(true_labels).intersection(set(pred_labels)) for true_labels, pred_labels in zip(y_true, y_pred)]
    union = [set(true_labels).union(set(pred_labels)) for true_labels, pred_labels in zip(y_true, y_pred)]

    # calculate the Jaccard similarity score for each sample
    jaccard_scores = [len(intersect) / len(uni) if len(uni) > 0 else 0 for intersect, uni in zip(intersection, union)]

    # return the average Jaccard similarity score over all samples
    return sum(jaccard_scores) / len(jaccard_scores)


# function to calculate the F1 score for a single label
def f1_score_macro(y_true, y_pred):
    tp = len(set(y_true).intersection(set(y_pred)))
    fp = len(set(y_pred)) - tp
    fn = len(set(y_true)) - tp
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# function to calculate the macro F1 score
def macro_f1_score_mlec(y_true, y_pred):
    # calculate the F1 score for each label
    f1_scores = [f1_score_macro(true_labels, pred_labels) for true_labels, pred_labels in zip(y_true, y_pred)]

    # calculate the macro F1 score by averaging over labels
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return macro_f1


# function to calculate the F1 score for all labels
def f1_score_micro(y_true, y_pred):
    tp = sum([len(set(true_labels).intersection(set(pred_labels))) for true_labels, pred_labels in zip(y_true, y_pred)])
    fp = sum([len(set(pred_labels).difference(set(true_labels))) for true_labels, pred_labels in zip(y_true, y_pred)])
    fn = sum([len(set(true_labels).difference(set(pred_labels))) for true_labels, pred_labels in zip(y_true, y_pred)])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# function to calculate the micro F1 score
def micro_f1_score(y_true, y_pred):
    micro_f1 = f1_score_micro(y_true, y_pred)
    return micro_f1


def f1_score_mcec(y_true, y_pred, class_label):
    tp = sum([1 for true_label, pred_label in zip(y_true, y_pred) if
              true_label == class_label and pred_label == class_label])
    fp = sum([1 for true_label, pred_label in zip(y_true, y_pred) if
              true_label != class_label and pred_label == class_label])
    fn = sum([1 for true_label, pred_label in zip(y_true, y_pred) if
              true_label == class_label and pred_label != class_label])
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# function to calculate the macro F1 score
def macro_f1_score_mcec(y_true, y_pred, num_classes):
    # calculate the F1 score for each class
    f1_scores = [f1_score_mcec(y_true, y_pred, class_label) for class_label in num_classes]

    # calculate the macro F1 score by averaging over classes
    macro_f1 = sum(f1_scores) / len(f1_scores)

    return macro_f1


# function to calculate the accuracy score
def accuracy_score(y_true, y_pred):
    correct_predictions = sum([1 for true_label, pred_label in zip(y_true, y_pred) if true_label == pred_label])
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy


def precision_recall_score_mcec(y_true, y_pred, classes):
    num_classes = len(classes)
    tag2index = {tag: idx for idx, tag in enumerate(classes)}
    tp = [0] * num_classes  # true positives
    fp = [0] * num_classes  # false positives
    fn = [0] * num_classes  # false negatives

    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            tp[tag2index[y_true[i]]] += 1
        else:
            fp[tag2index[y_pred[i]]] += 1
            fn[tag2index[y_true[i]]] += 1

    sum_tp = sum(tp)
    sum_fp = sum(fp)
    sum_fn = sum(fn)

    precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) != 0 else 0
    recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) != 0 else 0

    return precision, recall


def calculate_metrics(goldList_mlec, goldList_mcec, predictions_mlec, predictions_mcec, task1, task2):
    if task1:
        jaccard = jaccard_similarity(goldList_mlec, predictions_mlec)
        micro_f1 = micro_f1_score(goldList_mlec, predictions_mlec)
        macro_f1 = macro_f1_score_mlec(goldList_mlec, predictions_mlec)
    else:
        jaccard, micro_f1, macro_f1 = 0, 0, 0

    if task2:
        macro_f1_mcec = macro_f1_score_mcec(goldList_mcec, predictions_mcec, set(predictions_mcec))
        accuracy = accuracy_score(goldList_mcec, predictions_mcec)
        precision, recall = precision_recall_score_mcec(goldList_mcec, predictions_mcec, set(predictions_mcec))
    else:
        macro_f1_mcec, accuracy, precision, recall = 0, 0, 0, 0

    return jaccard, micro_f1, macro_f1, macro_f1_mcec, accuracy, precision, recall


def score(input_dir, output_dir):
    # unzipped reference data is always in the 'ref' subdirectory
    goldList_mlec = read_mlec(os.path.join(input_dir, 'ref', 'gold_mlec.csv'))
    goldList_mcec = read_mcec(os.path.join(input_dir, 'ref', 'gold_mcec.csv'))

    mlec_submission_path = os.path.join(input_dir, 'res', 'predictions_MLEC.csv')
    mcec_submission_path = os.path.join(input_dir, 'res', 'predictions_MCEC.csv')

    task1, task2 = False, False
    if os.path.exists(mlec_submission_path):
        task1 = True
        predictedList_MLEC = read_mlec(mlec_submission_path)
        if len(goldList_mlec) != len(predictedList_MLEC):
            eprint("Number of labels is not aligned for MLEC!")
            sys.exit(1)
    else:
        predictedList_MLEC = [[0] * 1] * 1191

    if os.path.exists(mcec_submission_path):
        task2 = True
        predictedList_MCEC = read_mcec(mcec_submission_path)

        if len(goldList_mcec) != len(predictedList_MCEC):
            eprint("Number of labels is not aligned for MCEC!")
            sys.exit(1)
    else:
        predictedList_MCEC = [[0] * 11] * 1191

    jaccard, micro_f1, macro_f1, macro_f1_mcec, accuracy, precision, recall = \
        calculate_metrics(goldList_mlec, goldList_mcec, predictedList_MLEC, predictedList_MCEC, task1, task2)

    with open(os.path.join(output_dir, 'scores.txt'), 'w') as output_file:
        str_to_write = ''
        if task1:
            str_to_write += "Task1: Multi label accuracy: {0}\nMacro F1: {1}\nMicro F1: {2}\n".format(jaccard, micro_f1,
                                                                                                      macro_f1)
        if task2:
            str_to_write += "Task2: Macro F1-Score: {0}\nRecall: {1}\nPrecision: {2}\nAccuracy: {3}\n".format(
                macro_f1_mcec, recall, precision, accuracy)

        output_file.write(str_to_write)


def main():
    [_, input_dir, output_dir] = sys.argv
    score(input_dir, output_dir)


if __name__ == '__main__':
    main()
