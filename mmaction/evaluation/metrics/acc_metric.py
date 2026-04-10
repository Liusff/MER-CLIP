# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import datetime
import os

import mmengine
import numpy as np
import pandas
import torch
from mmengine.evaluator import BaseMetric

from mmaction.evaluation.functional.accuracy import (
    get_weighted_score, mean_average_precision,
    mean_class_accuracy,
    mmit_mean_average_precision, top_k_accuracy)
from mmaction.registry import METRICS
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

def to_tensor(value):
    """Convert value to torch.Tensor."""
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    elif isinstance(value, Sequence) and not mmengine.is_str(value):
        value = torch.tensor(value)
    elif not isinstance(value, torch.Tensor):
        raise TypeError(f'{type(value)} is not an available argument.')
    return value

def fine2coarse(x):
    if x <= 4:
        return 0
    elif 5 <= x <= 10:
        return 1
    elif 11 <= x <= 23:
        return 2
    elif 24 <= x <= 31:
        return 3
    elif 32 <= x <= 37:
        return 4
    elif 38 <= x <= 47:
        return 5
    else:
        return 6

fine2coarse_matrix = [
    (0, 4, 0),
    (5, 10, 1),
    (11, 23, 2),
    (24, 31, 3),
    (32, 37, 4),
    (38, 47, 5),
    (48, 51, 6)
]

def get_coarse_predictions(fine_scores):
    fine_scores = np.array(fine_scores)
    coarse_num_classes = 7
    coarse_scores = np.zeros((fine_scores.shape[0], coarse_num_classes))

    for start_idx, end_idx, coarse_idx in fine2coarse_matrix:
        # 取细粒度分数的相关部分，并计算它们的平均值
        fine_subset_scores = fine_scores[:, start_idx:end_idx + 1]
        mean_fine_score = fine_subset_scores.mean(axis=1)
        # 将计算的平均细粒度分数加到对应的粗粒度分数上
        coarse_scores[:, coarse_idx] += mean_fine_score

    coarse_predictions = np.exp(coarse_scores) / np.sum(np.exp(coarse_scores), axis=1, keepdims=True)
    return coarse_predictions

@METRICS.register_module()
class AccMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            result['filename'] = data_sample['filename']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = self.calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            filenames = [x['filename'].split('/')[-1] for x in results]
            return self.calculate(preds, labels, filenames)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]],
                  filenames: List[str]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)

        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                fine_top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, fine_top_k_acc):
                    eval_results[f'fine_top{k}_acc'] = acc
        
        f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro=self.lv_evaluation(preds,labels,filenames)
        eval_results['F1_mean'] = f1_mean
        eval_results['F1_fine_mean'] = f1_fine_mean
        eval_results['F1_coarse_mean'] = f1_coarse_mean
        eval_results['fine_f1_micro'] = fine_f1_micro
        eval_results['fine_f1_macro'] = fine_f1_macro
        eval_results['coarse_f1_micro'] = coarse_f1_micro
        eval_results['coarse_f1_macro'] = coarse_f1_macro
        
        return eval_results
    def lv_evaluation(self,predictions,labels,filenames):
        coarse_predictions = get_coarse_predictions(predictions)
        predictions = np.argsort(predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_preds = np.argsort(coarse_predictions, axis=1)[:, -1::-1][:, 0].tolist()
        #coarse_preds = [fine2coarse(lv2id) for lv2id in predictions]
        coarse_labels = [fine2coarse(lv2id) for lv2id in labels]
        fine_f1_micro = f1_score(labels, predictions, average='micro')
        fine_f1_macro = f1_score(labels, predictions, average='macro')
        coarse_f1_micro = f1_score(coarse_labels, coarse_preds, average='micro')
        coarse_f1_macro = f1_score(coarse_labels, coarse_preds, average='macro')
        f1_mean = (fine_f1_macro + coarse_f1_macro + coarse_f1_micro + fine_f1_micro) / 4.0
        f1_fine_mean = (fine_f1_macro + fine_f1_micro) / 2.0
        f1_coarse_mean = (coarse_f1_macro + coarse_f1_micro) / 2.0
        '''
        result_df = pandas.DataFrame({'id':filenames, 'gt_label_1':coarse_preds, 'gt_label_2':predictions})
        result_df.to_csv('predictions_csv/prediction.csv',index=False)
        '''
        return f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro

@METRICS.register_module()
class AccMetric_test(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None,
                 path: str = "predictions_csv/") -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options
        self.path = path

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            result['filename'] = data_sample['filename']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = self.calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            filenames = [x['filename'].split('/')[-1] for x in results]
            return self.calculate(preds, labels, filenames)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]],
                  filenames: List[str]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)

        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                fine_top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, fine_top_k_acc):
                    eval_results[f'fine_top{k}_acc'] = acc
        
        f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro=self.lv_evaluation(preds,labels,filenames)
        
        eval_results['F1_mean'] = f1_mean
        eval_results['F1_fine_mean'] = f1_fine_mean
        eval_results['F1_coarse_mean'] = f1_coarse_mean
        eval_results['fine_f1_micro'] = fine_f1_micro
        eval_results['fine_f1_macro'] = fine_f1_macro
        eval_results['coarse_f1_micro'] = coarse_f1_micro
        eval_results['coarse_f1_macro'] = coarse_f1_macro
        
        return eval_results
    def lv_evaluation(self,predictions,labels,filenames):
        coarse_predictions = get_coarse_predictions(predictions)
        predictions = np.argsort(predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_preds = np.argsort(coarse_predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_labels = [fine2coarse(lv2id) for lv2id in labels]
        fine_f1_micro = f1_score(labels, predictions, average='micro')
        fine_f1_macro = f1_score(labels, predictions, average='macro')
        coarse_f1_micro = f1_score(coarse_labels, coarse_preds, average='micro')
        coarse_f1_macro = f1_score(coarse_labels, coarse_preds, average='macro')
        f1_mean = (fine_f1_macro + coarse_f1_macro + coarse_f1_micro + fine_f1_micro) / 4.0
        f1_fine_mean = (fine_f1_macro + fine_f1_micro) / 2.0
        f1_coarse_mean = (coarse_f1_macro + coarse_f1_micro) / 2.0

        result_df = pandas.DataFrame({'id':filenames, 'gt_label_1':coarse_preds, 'gt_label_2':predictions})
        current_time = datetime.datetime.now()
        # 格式化时间为字符串，例如 '2024-06-12_14-47-02'
        time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
        # 生成文件名
        result_path = os.path.join(self.path, f"test_{time_str}.csv")
        result_df.to_csv(result_path, index=False)

        return f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro

@METRICS.register_module()
class AccMetric_bodyaction(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None,
                 print_result: bool = False,
                 path: str = "predictions_csv/") -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options
        self.path = path
        self.print_result = print_result

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            coarse_pred = data_sample['pred_coarse_score']
            result['filename'] = data_sample['filename']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()
            
            if isinstance(coarse_pred, dict):
                for item_name, coarse_score in pred.items():
                    coarse_pred[item_name] = coarse_score.cpu().numpy()
            else:
                coarse_pred = coarse_pred.cpu().numpy()

            result['pred'] = pred
            result['coarse_pred'] = coarse_pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = self.calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            coarse_preds = [x['coarse_pred'] for x in results]
            filenames = [x['filename'].split('/')[-1] for x in results]
            return self.calculate(preds, coarse_preds, labels, filenames)

    def calculate(self, preds: List[np.ndarray], coarse_preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]],
                  filenames: List[str]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)

        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))

                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')

                if isinstance(topk, int):
                    topk = (topk, )

                fine_top_k_acc = top_k_accuracy(preds, labels, topk)
                for k, acc in zip(topk, fine_top_k_acc):
                    eval_results[f'fine_top{k}_acc'] = acc
        
        f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro=self.lv_evaluation(preds, coarse_preds, labels,filenames)
        eval_results['F1_mean'] = f1_mean
        eval_results['F1_fine_mean'] = f1_fine_mean
        eval_results['F1_coarse_mean'] = f1_coarse_mean
        eval_results['fine_f1_micro'] = fine_f1_micro
        eval_results['fine_f1_macro'] = fine_f1_macro
        eval_results['coarse_f1_micro'] = coarse_f1_micro
        eval_results['coarse_f1_macro'] = coarse_f1_macro
        
        return eval_results
    def lv_evaluation(self,predictions, coarse_predictions, labels,filenames):

        predictions = np.argsort(predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_preds = np.argsort(coarse_predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_labels = [fine2coarse(lv2id) for lv2id in labels]
        fine_f1_micro = f1_score(labels, predictions, average='micro')
        fine_f1_macro = f1_score(labels, predictions, average='macro')
        coarse_f1_micro = f1_score(coarse_labels, coarse_preds, average='micro')
        coarse_f1_macro = f1_score(coarse_labels, coarse_preds, average='macro')
        f1_mean = (fine_f1_macro + coarse_f1_macro + coarse_f1_micro + fine_f1_micro) / 4.0
        f1_fine_mean = (fine_f1_macro + fine_f1_micro) / 2.0
        f1_coarse_mean = (coarse_f1_macro + coarse_f1_micro) / 2.0
        
        if self.print_result:
            result_df = pandas.DataFrame({'id':filenames, 'gt_label_1':coarse_preds, 'gt_label_2':predictions})
            current_time = datetime.datetime.now()
            # 格式化时间为字符串，例如 '2024-06-12_14-47-02'
            time_str = current_time.strftime("%Y-%m-%d_%H-%M-%S")
            # 生成文件名
            result_path = os.path.join(self.path, f"test_{time_str}.csv")
            result_df.to_csv(result_path, index=False)

        return f1_mean, f1_fine_mean, f1_coarse_mean, fine_f1_micro, fine_f1_macro, coarse_f1_micro, coarse_f1_macro

@METRICS.register_module()
class AccMetric_body(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'top_k_accuracy', 'mean_class_accuracy'),
                 collect_device: str = 'cpu',
                 metric_options: Optional[Dict] = dict(
                     top_k_accuracy=dict(topk=(1, 5))),
                 prefix: Optional[str] = None) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        # coco evaluation metrics
        for metric in metrics:
            assert metric in [
                'top_k_accuracy', 'mean_class_accuracy',
                'mmit_mean_average_precision', 'mean_average_precision'
            ]

        self.metrics = metrics
        self.metric_options = metric_options
        
    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        # Ad-hoc for RGBPoseConv3D
        if isinstance(results[0]['pred'], dict):

            for item_name in results[0]['pred'].keys():
                preds = [x['pred'][item_name] for x in results]
                eval_result = self.calculate(preds, labels)
                eval_results.update(
                    {f'{item_name}_{k}': v
                     for k, v in eval_result.items()})

            if len(results[0]['pred']) == 2 and \
                    'rgb' in results[0]['pred'] and \
                    'pose' in results[0]['pred']:

                rgb = [x['pred']['rgb'] for x in results]
                pose = [x['pred']['pose'] for x in results]

                preds = {
                    '1:1': get_weighted_score([rgb, pose], [1, 1]),
                    '2:1': get_weighted_score([rgb, pose], [2, 1]),
                    '1:2': get_weighted_score([rgb, pose], [1, 2])
                }
                for k in preds:
                    eval_result = self.calculate(preds[k], labels)
                    eval_results.update({
                        f'RGBPose_{k}_{key}': v
                        for key, v in eval_result.items()
                    })
            return eval_results

        # Simple Acc Calculation
        else:
            preds = [x['pred'] for x in results]
            return self.calculate(preds, labels)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        metric_options = copy.deepcopy(self.metric_options)

        for metric in self.metrics:
            if metric == 'top_k_accuracy':
                
                acc = top_k_accuracy(preds, labels, (1,))
                eval_results[f'body_top1_acc'] = acc
        
        coarse_f1_micro, coarse_f1_macro, f1_mean = self.lv_evaluation(preds,labels)
        eval_results['coarse_f1_micro'] = coarse_f1_micro
        eval_results['coarse_f1_macro'] = coarse_f1_macro
        eval_results['F1_mean'] = f1_mean
        
        return eval_results
    def lv_evaluation(self,predictions,labels):
        predictions = np.argsort(predictions, axis=1)[:, -1::-1][:, 0].tolist()
        coarse_f1_micro = f1_score(labels, predictions, average='micro')
        coarse_f1_macro = f1_score(labels, predictions, average='macro')
        f1_mean = (coarse_f1_micro + coarse_f1_macro) / 2.0
        
        return coarse_f1_micro, coarse_f1_macro, f1_mean

def confusionMatrix_def(gt, pred, show=False):
    TN, FP, FN, TP = confusion_matrix(gt, pred).ravel()
    f1_score = (2 * TP) / (2 * TP + FP + FN)
    num_samples = len([x for x in gt if x == 1])
    average_recall = TP / num_samples
    return f1_score, average_recall


@METRICS.register_module()
class MEMetric(BaseMetric):
    """Accuracy evaluation metric."""
    default_prefix: Optional[str] = 'acc'

    def __init__(self,
                 metric_list: Optional[Union[str, Tuple[str]]] = (
                     'uf1', 'uar'),
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 save_results: bool = False,
                 save_path: str = None,
                 label_map: dict = {"0": "anger", "1": "contempt", "2": "disgust", "3": "fear", "4": "happiness", "5": "sadness", "6": "surprise"}) -> None:

        # TODO: fix the metric_list argument with a better one.
        # `metrics` is not a safe argument here with mmengine.
        # we have to replace it with `metric_list`.
        super().__init__(collect_device=collect_device, prefix=prefix)
        if not isinstance(metric_list, (str, tuple)):
            raise TypeError('metric_list must be str or tuple of str, '
                            f'but got {type(metric_list)}')

        if isinstance(metric_list, str):
            metrics = (metric_list, )
        else:
            metrics = metric_list

        self.label_map = label_map
        self.metrics = metrics
        self.save_results = save_results
        self.save_path = save_path

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of data samples and data_samples. The processed
        results should be stored in ``self.results``, which will be used to
        compute the metrics when all batches have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        data_samples = copy.deepcopy(data_samples)
        for data_sample in data_samples:
            #print("data sample: ", data_sample)
            #print("data_sample[0]: ", data_sample[0])
            result = dict()
            pred = data_sample['pred_score']
            label = data_sample['gt_label']
            result['filename'] = data_sample['filename']

            # Ad-hoc for RGBPoseConv3D
            if isinstance(pred, dict):
                for item_name, score in pred.items():
                    pred[item_name] = score.cpu().numpy()
            else:
                pred = pred.cpu().numpy()

            result['pred'] = pred
            if label.size(0) == 1:
                # single-label
                result['label'] = label.item()
            else:
                # multi-label
                result['label'] = label.cpu().numpy()
            self.results.append(result)

    def compute_metrics(self, results: List) -> Dict:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        labels = [x['label'] for x in results]

        eval_results = dict()
        
        # Simple Acc Calculation
        preds = [x['pred'] for x in results]
        filenames = [x['filename'].split('/')[-1] for x in results]
        return self.calculate(preds, labels, filenames)

    def calculate(self, preds: List[np.ndarray],
                  labels: List[Union[int, np.ndarray]],
                  filenames: List[str]) -> Dict:
        """Compute the metrics from processed results.

        Args:
            preds (list[np.ndarray]): List of the prediction scores.
            labels (list[int | np.ndarray]): List of the labels.

        Returns:
            dict: The computed metrics. The keys are the names of the metrics,
            and the values are corresponding results.
        """
        eval_results = OrderedDict()
        predictions = np.argsort(preds, axis=1)[:, -1::-1][:, 0].tolist()
        uf1, uar = self.recognition_evaluation(labels, predictions, filenames)
        acc = accuracy_score(labels, predictions, normalize=True, sample_weight=None)
        eval_results['UF1'] = uf1
        eval_results['UAR'] = uar
        eval_results['ACC'] = acc
        eval_results['pred'] = [self.label_map[str(x)] for x in predictions[:10]]
        eval_results['gt'] = [self.label_map[str(x)] for x in labels[:10]]
        if self.save_results:
            result_df = pandas.DataFrame({'id':filenames, 'pred':predictions, 'gt':labels})
            result_df.to_csv(self.save_path, mode='a', header=False, index=False)

        return eval_results

    def recognition_evaluation(self, final_gt, final_pred, filenames):
        unique_elements_gt = np.unique(final_gt)
        unique_elements_pred = np.unique(final_pred)
        # 如果数组中只有一个唯一元素，则表示只包含一类数字
        if len(unique_elements_gt) == 1 and len(unique_elements_pred) == 1 and unique_elements_gt[0] == unique_elements_pred[0]:
            UF1 = 1
            UAR = 1
            return UF1, UAR 

        label_dict = {v: int(k) for k, v in self.label_map.items()}
        # Display recognition result
        f1_list = []
        ar_list = []
        try:
            for emotion, emotion_index in label_dict.items():
                gt_recog = [1 if x == emotion_index else 0 for x in final_gt]
                pred_recog = [1 if x == emotion_index else 0 for x in final_pred]
                try:
                    f1_recog, ar_recog = confusionMatrix_def(gt_recog, pred_recog)
                    f1_list.append(f1_recog)
                    ar_list.append(ar_recog)
                except Exception as e:
                    pass
            UF1 = np.mean(f1_list)
            UAR = np.mean(ar_list)
            return UF1, UAR
        except:
            return ' ',' '


@METRICS.register_module()
class ConfusionMatrix(BaseMetric):
    r"""A metric to calculate confusion matrix for single-label tasks.

    Args:
        num_classes (int, optional): The number of classes. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.

    Examples:

        1. The basic usage.

        >>> import torch
        >>> from mmaction.evaluation import ConfusionMatrix
        >>> y_pred = [0, 1, 1, 3]
        >>> y_true = [0, 2, 1, 3]
        >>> ConfusionMatrix.calculate(y_pred, y_true, num_classes=4)
        tensor([[1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]])
        >>> # plot the confusion matrix
        >>> import matplotlib.pyplot as plt
        >>> y_score = torch.rand((1000, 10))
        >>> y_true = torch.randint(10, (1000, ))
        >>> matrix = ConfusionMatrix.calculate(y_score, y_true)
        >>> ConfusionMatrix().plot(matrix)
        >>> plt.show()

        2. In the config file

        .. code:: python

            val_evaluator = dict(type='ConfusionMatrix')
            test_evaluator = dict(type='ConfusionMatrix')
    """  # noqa: E501
    default_prefix = 'confusion_matrix'

    def __init__(self,
                 num_classes: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device, prefix)

        self.num_classes = num_classes

    def process(self, data_batch, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            pred_scores = data_sample.get('pred_score')
            gt_label = data_sample['gt_label']
            if pred_scores is not None:
                pred_label = pred_scores.argmax(dim=0, keepdim=True)
                self.num_classes = pred_scores.size(0)
            else:
                pred_label = data_sample['pred_label']

            self.results.append({
                'pred_label': pred_label,
                'gt_label': gt_label
            })

    def compute_metrics(self, results: list) -> dict:
        pred_labels = []
        gt_labels = []
        for result in results:
            pred_labels.append(result['pred_label'])
            gt_labels.append(result['gt_label'])
        confusion_matrix = ConfusionMatrix.calculate(
            torch.cat(pred_labels),
            torch.cat(gt_labels),
            num_classes=self.num_classes)
        return {'result': confusion_matrix}

    @staticmethod
    def calculate(pred, target, num_classes=None) -> dict:
        """Calculate the confusion matrix for single-label task.

        Args:
            pred (torch.Tensor | np.ndarray | Sequence): The prediction
                results. It can be labels (N, ), or scores of every
                class (N, C).
            target (torch.Tensor | np.ndarray | Sequence): The target of
                each prediction with shape (N, ).
            num_classes (Optional, int): The number of classes. If the ``pred``
                is label instead of scores, this argument is required.
                Defaults to None.

        Returns:
            torch.Tensor: The confusion matrix.
        """
        pred = to_tensor(pred)
        target_label = to_tensor(target).int()

        assert pred.size(0) == target_label.size(0), \
            f"The size of pred ({pred.size(0)}) doesn't match "\
            f'the target ({target_label.size(0)}).'
        assert target_label.ndim == 1

        if pred.ndim == 1:
            assert num_classes is not None, \
                'Please specify the `num_classes` if the `pred` is labels ' \
                'intead of scores.'
            pred_label = pred
        else:
            num_classes = num_classes or pred.size(1)
            pred_label = torch.argmax(pred, dim=1).flatten()

        with torch.no_grad():
            indices = num_classes * target_label + pred_label
            matrix = torch.bincount(indices, minlength=num_classes**2)
            matrix = matrix.reshape(num_classes, num_classes)

        return matrix

    @staticmethod
    def plot(confusion_matrix: torch.Tensor,
             include_values: bool = False,
             cmap: str = 'viridis',
             classes: Optional[List[str]] = None,
             colorbar: bool = True,
             show: bool = True):
        """Draw a confusion matrix by matplotlib.

        Modified from `Scikit-Learn
        <https://github.com/scikit-learn/scikit-learn/blob/dc580a8ef/sklearn/metrics/_plot/confusion_matrix.py#L81>`_

        Args:
            confusion_matrix (torch.Tensor): The confusion matrix to draw.
            include_values (bool): Whether to draw the values in the figure.
                Defaults to False.
            cmap (str): The color map to use. Defaults to use "viridis".
            classes (list[str], optional): The names of categories.
                Defaults to None, which means to use index number.
            colorbar (bool): Whether to show the colorbar. Defaults to True.
            show (bool): Whether to show the figure immediately.
                Defaults to True.
        """  # noqa: E501
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 10))

        num_classes = confusion_matrix.size(0)

        im_ = ax.imshow(confusion_matrix, interpolation='nearest', cmap=cmap)
        text_ = None
        cmap_min, cmap_max = im_.cmap(0), im_.cmap(1.0)

        if include_values:
            text_ = np.empty_like(confusion_matrix, dtype=object)

            # print text with appropriate color depending on background
            thresh = (confusion_matrix.max() + confusion_matrix.min()) / 2.0

            for i, j in product(range(num_classes), range(num_classes)):
                color = cmap_max if confusion_matrix[i,
                                                     j] < thresh else cmap_min

                text_cm = format(confusion_matrix[i, j], '.2g')
                text_d = format(confusion_matrix[i, j], 'd')
                if len(text_d) < len(text_cm):
                    text_cm = text_d

                text_[i, j] = ax.text(
                    j, i, text_cm, ha='center', va='center', color=color)

        display_labels = classes or np.arange(num_classes)

        if colorbar:
            fig.colorbar(im_, ax=ax)
        ax.set(
            xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=display_labels,
            yticklabels=display_labels,
            ylabel='True label',
            xlabel='Predicted label',
        )
        ax.invert_yaxis()
        ax.xaxis.tick_top()

        ax.set_ylim((num_classes - 0.5, -0.5))
        # Automatically rotate the x labels.
        fig.autofmt_xdate(ha='center')

        if show:
            plt.show()
        return fig
