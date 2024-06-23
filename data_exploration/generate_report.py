import pandas as pd
import sys

sys.path.append('..')  # Adjust the path based on your project structure
import json
import re
import ast
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, RocCurveDisplay,  average_precision_score
import matplotlib.pyplot as plt
import os.path as osp
from glob import glob
from miga.const import ID2LABELS_SMG_SHORT as ID2LABELS
from typing import List, Union, Tuple
from utils import load_and_parse_txt

#
# def generate_report(path_to_folder: str, path_dataset_folder: str, feature_names: List[str] = None) -> None:
#     if feature_names is None:
#         # Get feature names by loading the weights folder
#         path_to_weights = osp.join(path_dataset_folder, 'weights.json')
#         with open(path_to_weights, 'r') as json_file:
#             positive_weights_dict = json.load(json_file)
#
#         feature_names = [entry['class'] for entry in positive_weights_dict['data']]
#
#     path_to_log = glob(osp.join(path_to_folder, '*.txt'))
#     path_to_dataset = osp.join(path_dataset_folder, 'val.csv')
#
#     # Get names of file names from the csv file
#     df_csv = pd.read_csv(path_to_dataset)
#     file_names = df_csv['filenames'].tolist()
#     metadata = df_csv['metadata'].tolist()
#     df_all, logit_columns, pred_columns, gt_columns = load_and_parse_txt(path_to_log, feature_names, file_names)
#
#     # If there are multiple validation files, get mean pred per file
#     df = df_all.drop(columns='log_name').groupby('filenames').mean()
#     logit_columns = logit_columns[0]
#     pred_columns = pred_columns[0]
#     gt_columns = gt_columns[0]
#
#     # df['side'] = df.index.str.split('/').str[-2]
#     # df['camera_name'] = df.index.str.split('/').str[-3]
#     # df['view'] = df['camera_name'].apply(lambda x: 'top' if x == 'Cam3' else 'front' if x == 'Cam4' else None)
#
#     # Save concatenated and averaged files
#     df.to_csv(osp.join(path_to_folder, 'raw_averaged_test_results.csv'))
#     df_all.to_csv(osp.join(path_to_folder, 'raw_test_results.csv'))
#
#     # Summarize metrics using classification report
#     def generate_report_helper(df):
#         gt = df[gt_columns]
#         predictions = df[pred_columns]
#         logits = df[logit_columns]
#
#         classification_rep = classification_report(gt, predictions, output_dict=True, target_names=feature_names)
#         df_report = pd.DataFrame.from_dict(classification_rep).transpose()
#
#         def id_to_labels(x):
#             try:
#                 x = int(x)
#             except:
#                 return x
#             return ID2LABELS[x + 1]
#
#         df_report['labels'] = df_report.reset_index()['index'].apply(id_to_labels)
#
#
#         # add auc
#         roc_auc_scores = {}
#         roc_metrics = {}
#
#         feature_names_str = [str(a) for a in feature_names]
#         for column in feature_names_str:
#             true_label = gt['gt-' + column]
#             pred_prob = logits['logit-' + column]
#             roc_auc = roc_auc_score(true_label, pred_prob)
#             roc_auc_scores[column] = roc_auc
#             fpr, tpr, thresholds = roc_curve(true_label, pred_prob)
#             roc_metrics[column] = dict(
#                 fpr=fpr,
#                 tpr=tpr,
#                 thresholds=thresholds,
#                 auc=roc_auc
#             )
#
#         # Display ROC-AUC scores for each label
#         for label, roc_auc in roc_auc_scores.items():
#             print(f'ROC-AUC for {label}: {roc_auc}')
#
#         return df_report, logits, gt
#
#     df_report, logits, gt = generate_report_helper(df)
#
#     # Save report
#     df_report.to_csv(osp.join(path_to_folder, 'classification_report.csv'))
#
#
# def generate_report(df_in, th=None):
#     def get_scores(metric_name, metric_function):
#
#         average_metric = dict()
#         for class_name in y_true_df.columns:
#             y_true = gt[class_name].values
#             y_pred = logits[class_name.replace('gt', 'logit')].values
#             ap = metric_function(y_true, y_pred)
#             average_metric[int(class_name.replace('gt-', ''))] = ap
#
#         avg_types = ['micro', 'samples', 'weighted', 'macro']
#         metric_df = dict()
#         for avg_type in avg_types:
#             metric_df[avg_type + ' avg'] = metric_function(gt, logits, average=avg_type)
#         average_metric.update(metric_df)
#         average_metric_df = pd.DataFrame(average_metric.values(), index=average_metric.keys(), columns=[metric_name])
#         return average_metric_df
#
#     df = df_in.copy()
#     if th is not None:
#         if isinstance(th, float):
#             th_dict = {k: th for k in range(len(feature_names))}
#         elif isinstance(th, dict):
#             th_dict = th
#
#         for k, v in th_dict.items():
#             df['pred-' + str(k)] = df['logit-' + str(k)] >= v
#     else:
#         th_dict = {k: np.nan for k in range(len(feature_names))}
#
#     gt = df[gt_columns]
#     predictions = df[pred_columns]
#     logits = df[logit_columns]
#
#     classification_rep = classification_report(gt, predictions, output_dict=True, target_names=feature_names)
#     df_report = pd.DataFrame.from_dict(classification_rep).transpose()
#
#     def id_to_labels(x):
#         try:
#             x = int(x)
#         except:
#             return x
#         return ID2LABELS[x + 1]
#
#     df_ap = get_scores(metric_name='AP', metric_function=average_precision_score)
#     df_auc = get_scores(metric_name='AUC', metric_function=roc_auc_score)
#     df_th = pd.DataFrame(th_dict.values(), index=th_dict.keys(), columns=['threshold'])
#     df_report = df_report.join(df_ap).join(df_auc).join(df_th)
#
#     df_report['labels'] = df_report.reset_index()['index'].apply(id_to_labels)
#
#     # add auc
#     # Calculate ROC-AUC for each label
#     roc_auc_scores = {}
#     roc_metrics = {}
#
#     feature_names_str = [str(a) for a in feature_names]
#     for column in feature_names_str:
#         true_label = gt['gt-' + column]
#         pred_prob = logits['logit-' + column]
#         roc_auc = roc_auc_score(true_label, pred_prob)
#         roc_auc_scores[int(column)] = roc_auc
#         fpr, tpr, thresholds = roc_curve(true_label, pred_prob)
#         roc_metrics[column] = dict(
#             fpr=fpr,
#             tpr=tpr,
#             thresholds=thresholds,
#             auc=roc_auc
#         )
#
#     return df_report, logits, gt


def generate_combined_report(path_to_folder: str, path_dataset_folder: str, feature_names: List[str] = None, th=0.5,
                             id_to_labels_map: Union[List, dict, None]=None ):
    if feature_names is None:
        # Get feature names by loading the weights folder
        path_to_weights = osp.join(path_dataset_folder, 'weights.json')
        with open(path_to_weights, 'r') as json_file:
            positive_weights_dict = json.load(json_file)

        feature_names = [entry['class'] for entry in positive_weights_dict['data']]

    if id_to_labels_map is None:
        id_to_labels_map = {a:b for a, b in enumerate(feature_names)}

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

    # Save concatenated and averaged files
    df.to_csv(osp.join(path_to_folder, 'raw_averaged_test_results.csv'))
    df_all.to_csv(osp.join(path_to_folder, 'raw_test_results.csv'))

    # Threshold adjustment
    if th is not None:
        if isinstance(th, float):
            th_dict = {k: th for k in feature_names}
        elif isinstance(th, dict):
            th_dict = th
        else:
            raise TypeError('th is not a float or a dict')
        for k, v in th_dict.items():
            df['pred-' + k] = df['logit-' + k] >= v
    else:
        th_dict = {k: np.nan for k in feature_names}

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
        return id_to_labels_map[x + 1]



    # df_report['labels'] = df_report.reset_index()['index'].apply(id_to_labels)


    # Calculate additional metrics
    def get_scores(metric_name, metric_function):
        average_metric = dict()
        for class_name in gt.columns:
            y_true = gt[class_name].values
            y_pred = logits[class_name.replace('gt', 'logit')].values
            ap = metric_function(y_true, y_pred)
            average_metric[class_name.replace('gt-', '')] = ap

        avg_types = ['micro', 'samples', 'weighted', 'macro']
        metric_df = dict()
        for avg_type in avg_types:
            metric_df[avg_type + ' avg'] = metric_function(gt, logits, average=avg_type)
        average_metric.update(metric_df)
        average_metric_df = pd.DataFrame(average_metric.values(), index=average_metric.keys(), columns=[metric_name])
        return average_metric_df

    df_ap = get_scores(metric_name='AP', metric_function=average_precision_score)
    df_auc = get_scores(metric_name='AUC', metric_function=roc_auc_score)
    df_th = pd.DataFrame(th_dict.values(), index=th_dict.keys(), columns=['threshold'])
    df_report = df_report.join(df_ap).join(df_auc).join(df_th)

    # Calculate ROC-AUC for each label
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

    # Save report
    df_report = df_report.reset_index()
    df_report = df_report.rename(columns={'index':'labels'})
    df_report.to_csv(osp.join(path_to_folder, 'classification_report.csv'))

    # Display ROC-AUC scores for each label
    for label, roc_auc in roc_auc_scores.items():
        print(f'ROC-AUC for {label}: {roc_auc}')

    return df_report, logits, gt

if __name__ == '__main__':
    path_to_folder = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\mac_multi\eval_93'
    path_dataset_folder = r'D:\Project-mpg microgesture\human_micro_gesture_classifier\experiments\mac_multi\dataset'
    feature_names = None
    df_report, _, _, = generate_combined_report(path_to_folder, path_dataset_folder, feature_names)
    print(df_report)
    print('Done...')
