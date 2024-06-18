import pandas as pd
import sys

sys.path.append('..')  # Adjust the path based on your project structure
import json
import re
import ast
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, RocCurveDisplay
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
from miga.const import ID2LABELS_SMG_SHORT as ID2LABELS
from typing import List, Union, Tuple
from utils import load_and_parse_txt


def generate_report(path_to_folder: str, path_dataset_folder: str, feature_names: List[str] = None) -> None:
    if feature_names is None:
        # Get feature names by loading the weights folder
        path_to_weights = osp.join(path_dataset_folder, 'weights.json')
        with open(path_to_weights, 'r') as json_file:
            positive_weights_dict = json.load(json_file)

        feature_names = [entry['class'] for entry in positive_weights_dict['data']]

    path_to_log = glob(osp.join(path_to_folder, '*.txt'))
    path_to_dataset = osp.join(path_dataset_folder, 'val.csv')

    # Get names of file names from the csv file
    df_csv = pd.read_csv(path_to_dataset)
    file_names = df_csv['filenames'].tolist()
    metadata = df_csv['metadata'].tolist()
    df_all, logit_columns, pred_columns, gt_columns = load_and_parse_txt(path_to_log, feature_names, file_names)

    # If there are multiple validation files, get mean pred per file
    df = df_all.drop(columns='log_name').groupby('filenames').mean()
    logit_columns = logit_columns[0]
    pred_columns = pred_columns[0]
    gt_columns = gt_columns[0]

    df['side'] = df.index.str.split('/').str[-2]
    df['camera_name'] = df.index.str.split('/').str[-3]
    df['view'] = df['camera_name'].apply(lambda x: 'top' if x == 'Cam3' else 'front' if x == 'Cam4' else None)

    # Save concatenated and averaged files
    df.to_csv(osp.join(path_to_folder, 'raw_averaged_test_results.csv'))
    df_all.to_csv(osp.join(path_to_folder, 'raw_test_results.csv'))

    # Summarize metrics using classification report
    def generate_report(df):
        gt = df[gt_columns]
        predictions = df[pred_columns]
        logits = df[logit_columns]

        classification_rep = classification_report(gt, predictions, output_dict=True, target_names=feature_names)
        df_report = pd.DataFrame.from_dict(classification_rep).transpose()

        def id_to_labels(x):
            try:
                x = int(x)
            except:
                return x
            return ID2LABELS[x + 1]

        df_report['labels'] = df_report.reset_index()['index'].apply(id_to_labels)


        # add auc
        roc_auc_scores = {}
        roc_metrics = {}

        feature_names_str = [str(a) for a in feature_names]
        for column in feature_names_str:
            true_label = gt['gt-' + column]
            pred_prob = logits['logit-' + column]
            roc_auc = roc_auc_score(true_label, pred_prob)
            roc_auc_scores[column] = roc_auc
            fpr, tpr, thresholds = roc_curve(true_label, pred_prob)
            roc_metrics[column] = dict(
                fpr=fpr,
                tpr=tpr,
                thresholds=thresholds,
                auc=roc_auc
            )

        # Display ROC-AUC scores for each label
        for label, roc_auc in roc_auc_scores.items():
            print(f'ROC-AUC for {label}: {roc_auc}')

        return df_report, logits, gt

    df_report, logits, gt = generate_report(df)

    # Save report
    df_report.to_csv(osp.join(path_to_folder, 'classification_report.csv'))

