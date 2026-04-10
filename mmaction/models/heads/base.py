# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmaction.evaluation import top_k_accuracy
from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList

fine2coarse = [
    (0, 4, 0),
    (5, 10, 1),
    (11, 23, 2),
    (24, 31, 3),
    (32, 37, 4),
    (38, 47, 5),
    (48, 51, 6)
]

def deffine2coarse(x):
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
    elif 48 <= x <= 51:
        return 6
    else:
        return None

class AvgConsensus(nn.Module):
    """Average consensus module.

    Args:
        dim (int): Decide which dim consensus function to apply.
            Defaults to 1.
    """

    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the computation performed at every call."""
        return x.mean(dim=self.dim, keepdim=True)


class BaseHead(BaseModule, metaclass=ABCMeta):
    """Base class for head.

    All Head should subclass it.
    All subclass should overwrite:
    - :meth:`forward`, supporting to forward both for training and testing.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Defaults to ``dict(type='CrossEntropyLoss', loss_weight=1.0)``.
        multi_class (bool): Determines whether it is a multi-class
            recognition task. Defaults to False.
        label_smooth_eps (float): Epsilon used in label smooth.
            Reference: arxiv.org/abs/1906.02629. Defaults to 0.
        topk (int or tuple): Top-k accuracy. Defaults to ``(1, 5)``.
        average_clips (dict, optional): Config for averaging class
            scores over multiple clips. Defaults to None.
        init_cfg (dict, optional): Config to control the initialization.
            Defaults to None.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: Dict = dict(
                     type='CrossEntropyLoss', loss_weight=1.0),
                 multi_class: bool = False,
                 label_smooth_eps: float = 0.0,
                 topk: Union[int, Tuple[int]] = (1, 5),
                 average_clips: Optional[Dict] = None,
                 init_cfg: Optional[Dict] = None) -> None:
        super(BaseHead, self).__init__(init_cfg=init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.loss_cls = MODELS.build(loss_cls)
        self.multi_class = multi_class
        self.label_smooth_eps = label_smooth_eps
        self.average_clips = average_clips
        assert isinstance(topk, (int, tuple))
        if isinstance(topk, int):
            topk = (topk, )
        for _topk in topk:
            assert _topk > 0, 'Top-k should be larger than 0'
        self.topk = topk

    @abstractmethod
    def forward(self, x, **kwargs) -> ForwardResults:
        """Defines the computation performed at every call."""
        raise NotImplementedError

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Perform forward propagation of head and loss calculation on the
        features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        cls_scores, feat = self(feats, **kwargs)
        return self.loss_by_feat(cls_scores, data_samples)

    def loss_by_feat(self, cls_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
    
    def loss_by_2feats(self, cls_scores1: torch.Tensor, cls_scores2: torch.Tensor, beta: float,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        assert cls_scores1.shape == cls_scores2.shape
        cls_scores = beta * cls_scores1 + cls_scores2
        cls_scores = torch.softmax(cls_scores, dim=1)

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(cls_scores.device)
        labels = labels.squeeze()

        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and cls_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if cls_scores1.size() != labels.size():
            top_k_acc = top_k_accuracy(cls_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=cls_scores.device)
        if self.label_smooth_eps != 0:
            if cls_scores.size() != labels.size():
                labels = F.one_hot(labels, num_classes=self.num_classes)
            labels = ((1 - self.label_smooth_eps) * labels +
                      self.label_smooth_eps / self.num_classes)

        loss_cls = self.loss_cls(cls_scores, labels)
        # loss_cls may be dictionary or single tensor
        if isinstance(loss_cls, dict):
            losses.update(loss_cls)
        else:
            losses['loss_cls'] = loss_cls
        return losses
    
    def loss_by_finecoarse_feats(self, fine_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        self.coarse_num_classes = 7
        coarse_scores = torch.zeros((fine_scores.shape[0], self.coarse_num_classes)).to(fine_scores.device)

        for start_idx, end_idx, coarse_idx in fine2coarse:
            # 取细粒度分数的相关部分，并计算它们的平均值
            fine_subset_scores = fine_scores[:, start_idx:end_idx + 1]
            mean_fine_score = fine_subset_scores.mean(dim=1)
            # 将计算的平均细粒度分数加到对应的粗粒度分数上
            coarse_scores[:, coarse_idx] += mean_fine_score

        coarse_scores = torch.softmax(coarse_scores, dim=1)

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(fine_scores.device)
        labels = labels.squeeze()

        coarse_labels = torch.stack([torch.tensor(deffine2coarse(x)) for x in labels]).to(coarse_scores.device).squeeze()
        
        losses = dict()
        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)
        elif labels.dim() == 1 and labels.size()[0] == self.num_classes \
                and fine_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            labels = labels.unsqueeze(0)

        if coarse_labels.shape == torch.Size([]):
            coarse_labels = coarse_labels.unsqueeze(0)
        elif coarse_labels.dim() == 1 and coarse_labels.size()[0] == self.coarse_num_classes \
                and coarse_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            coarse_labels = coarse_labels.unsqueeze(0)

        if fine_scores.size() != labels.size():
            top_k_acc = top_k_accuracy(fine_scores.detach().cpu().numpy(),
                                       labels.detach().cpu().numpy(),
                                       self.topk)
            for k, a in zip(self.topk, top_k_acc):
                losses[f'top{k}_acc'] = torch.tensor(
                    a, device=fine_scores.device)

        fine_loss_cls = self.loss_cls(fine_scores, labels)
        coarse_loss_cls = self.loss_cls(coarse_scores, coarse_labels)
        # loss_cls may be dictionary or single tensor
        losses['loss_fine_cls'] = fine_loss_cls
        losses['loss_coarse_cls'] = coarse_loss_cls
        return losses
    
    def loss_by_coarsefeat(self, coarse_scores: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate the loss based on the features extracted by the head.

        Args:
            cls_scores (torch.Tensor): Classification prediction results of
                all class, has shape (batch_size, num_classes).
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
            dict: A dictionary of loss components.
        """
        self.coarse_num_classes = 7

        labels = [x.gt_label for x in data_samples]
        labels = torch.stack(labels).to(coarse_scores.device)
        labels = labels.squeeze()

        coarse_labels = torch.stack([torch.tensor(deffine2coarse(x)) for x in labels]).to(coarse_scores.device).squeeze()
        
        losses = dict()

        if coarse_labels.shape == torch.Size([]):
            coarse_labels = coarse_labels.unsqueeze(0)
        elif coarse_labels.dim() == 1 and coarse_labels.size()[0] == self.coarse_num_classes \
                and coarse_scores.size()[0] == 1:
            # Fix a bug when training with soft labels and batch size is 1.
            # When using soft labels, `labels` and `cls_score` share the same
            # shape.
            coarse_labels = coarse_labels.unsqueeze(0)

        coarse_loss_cls = self.loss_cls(coarse_scores, coarse_labels)
        # loss_cls may be dictionary or single tensor
        losses['loss_coarse_cls'] = coarse_loss_cls
        return losses

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores, pred_feat = self(feats, **kwargs)
        return self.predict_by_feat(cls_scores, data_samples), pred_feat 

    def predict_by_feat(self, cls_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)
        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        return data_samples
    
    def predict_by_coarsefeat(self, coarse_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = coarse_scores.shape[0] // len(data_samples)
        coarse_scores = self.average_clip(coarse_scores, num_segs=num_segs)
        pred_coarse_labels = coarse_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, coarse_score, pred_coarse_label in zip(data_samples, coarse_scores,
                                                  pred_coarse_labels):
            data_sample.set_pred_coarse_score(coarse_score)
            data_sample.set_pred_coarse_label(pred_coarse_label)
        return data_samples
    

    def predict_withclip(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]], clip_cls_scores: torch.Tensor, beta: float,
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation of head and predict recognition results
        on the features of the upstream network.

        Args:
            feats (torch.Tensor | tuple[torch.Tensor]): Features from
                upstream network.
            data_samples (list[:obj:`ActionDataSample`]): The batch
                data samples.

        Returns:
             list[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        cls_scores, feat = self(feats, **kwargs)
        return self.predict_by_feat_withclip(cls_scores, clip_cls_scores, beta, data_samples)

    def predict_by_feat_withclip(self, cls_scores: torch.Tensor, clip_cls_scores: torch.Tensor, beta: float,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = cls_scores.shape[0] // len(data_samples)
        cls_scores = self.average_clip(cls_scores, num_segs=num_segs)

        cls_scores = cls_scores + beta * clip_cls_scores

        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        return data_samples
    
    def predict_by_2feats(self, cls_scores1: torch.Tensor, cls_scores2: torch.Tensor, beta: float,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        assert cls_scores1.shape == cls_scores2.shape
        num_segs = cls_scores1.shape[0] // len(data_samples)
        cls_scores1 = self.average_clip(cls_scores1, num_segs=num_segs)
        cls_scores2 = self.average_clip(cls_scores2, num_segs=num_segs)

        cls_scores = beta * cls_scores1 + cls_scores2
        
        cls_scores = torch.softmax(cls_scores, dim=1)

        pred_labels = cls_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label in zip(data_samples, cls_scores,
                                                  pred_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
        return data_samples

    def predict_by_fused_finecoarse_feats(self, fine_scores: torch.Tensor, coarse_scores: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = fine_scores.shape[0] // len(data_samples)
        fine_scores = self.average_clip(fine_scores, num_segs=num_segs)
        coarse_scores = self.average_clip(coarse_scores, num_segs=num_segs)
        self.coarse_num_classes = 7

        fused_fine_scores = fine_scores.clone()
        fused_coarse_scores = coarse_scores.clone()

        for start_idx, end_idx, coarse_idx in fine2coarse:
            # 取对应的粗粒度分数
            coarse_subset_score = coarse_scores[:, coarse_idx].unsqueeze(1)
            # 将粗粒度分数加到相应的细粒度分数部分上
            fused_fine_scores[:, start_idx:end_idx + 1] += coarse_subset_score * 0.5
            # 取细粒度分数的相关部分，并计算它们的平均值
            fine_subset_scores = fine_scores[:, start_idx:end_idx + 1]
            mean_fine_score = fine_subset_scores.mean(dim=1)
            # 将计算的平均细粒度分数加到对应的粗粒度分数上
            fused_coarse_scores[:, coarse_idx] += mean_fine_score * 0.5

        fused_fine_scores = torch.softmax(fused_fine_scores, dim=1)
        fused_coarse_scores = torch.softmax(fused_coarse_scores, dim=1)
        
        pred_labels = fused_fine_scores.argmax(dim=-1, keepdim=True).detach()
        pred_coarse_labels = fused_coarse_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label, score_coarse, pred_coarse_label in zip(data_samples, fused_fine_scores, pred_labels,
                                                                                 fused_coarse_scores, pred_coarse_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
            data_sample.set_pred_coarse_score(score_coarse)
            data_sample.set_pred_coarse_label(pred_coarse_label)
        return data_samples
    
    def predict_by_fused_finecoarse_feats_withclip(self, fine_scores: torch.Tensor, coarse_scores: torch.Tensor, beta: float,
                         clip_cls_scores: torch.Tensor, data_samples: SampleList) -> SampleList:
        """Transform a batch of output features extracted from the head into
        prediction results.

        Args:
            cls_scores (torch.Tensor): Classification scores, has a shape
                (B*num_segs, num_classes)
            data_samples (list[:obj:`ActionDataSample`]): The
                annotation data of every samples. It usually includes
                information such as `gt_label`.

        Returns:
            List[:obj:`ActionDataSample`]: Recognition results wrapped
                by :obj:`ActionDataSample`.
        """
        num_segs = fine_scores.shape[0] // len(data_samples)
        fine_scores = self.average_clip(fine_scores, num_segs=num_segs)
        fine_scores = fine_scores + beta * clip_cls_scores

        coarse_scores = self.average_clip(coarse_scores, num_segs=num_segs)
        self.coarse_num_classes = 7

        fused_fine_scores = fine_scores.clone()
        fused_coarse_scores = coarse_scores.clone()

        for start_idx, end_idx, coarse_idx in fine2coarse:
            # 取对应的粗粒度分数
            coarse_subset_score = coarse_scores[:, coarse_idx].unsqueeze(1)
            # 将粗粒度分数加到相应的细粒度分数部分上
            fused_fine_scores[:, start_idx:end_idx + 1] += coarse_subset_score * 0.5
            # 取细粒度分数的相关部分，并计算它们的平均值
            fine_subset_scores = fine_scores[:, start_idx:end_idx + 1]
            mean_fine_score = fine_subset_scores.mean(dim=1)
            # 将计算的平均细粒度分数加到对应的粗粒度分数上
            fused_coarse_scores[:, coarse_idx] += mean_fine_score * 0.5

        fused_fine_scores = torch.softmax(fused_fine_scores, dim=1)
        fused_coarse_scores = torch.softmax(fused_coarse_scores, dim=1)
        
        pred_labels = fused_fine_scores.argmax(dim=-1, keepdim=True).detach()
        pred_coarse_labels = fused_coarse_scores.argmax(dim=-1, keepdim=True).detach()

        for data_sample, score, pred_label, score_coarse, pred_coarse_label in zip(data_samples, fused_fine_scores, pred_labels,
                                                                                 fused_coarse_scores, pred_coarse_labels):
            data_sample.set_pred_score(score)
            data_sample.set_pred_label(pred_label)
            data_sample.set_pred_coarse_score(score_coarse)
            data_sample.set_pred_coarse_label(pred_coarse_label)
        return data_samples
    
    
    def average_clip(self,
                     cls_scores: torch.Tensor,
                     num_segs: int = 1) -> torch.Tensor:
        """Averaging class scores over multiple clips.

        Using different averaging types ('score' or 'prob' or None,
        which defined in test_cfg) to computed the final averaged
        class score. Only called in test mode.

        Args:
            cls_scores (torch.Tensor): Class scores to be averaged.
            num_segs (int): Number of clips for each input sample.

        Returns:
            torch.Tensor: Averaged class scores.
        """

        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        batch_size = cls_scores.shape[0]
        cls_scores = cls_scores.view((batch_size // num_segs, num_segs) +
                                     cls_scores.shape[1:])

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores


    
