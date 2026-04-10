# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from mmaction.registry import MODELS
from .base import BaseWeightedLoss

MA_class_weight = [0.0275, 0.0133, 0.0018, 0.0008, 0.0139, 0.03, 0.0215, 0.0405, 0.0855, 0.114, 0.106, 0.0202, 0.01, 0.0084, 0.0037, 0.0073, 0.0068, 0.0316, 0.0077, 0.0246, 0.0131, 0.0316, 0.0332, 0.0353, 0.0212, 0.0059, 0.0082, 0.0086, 0.0053, 0.0205, 0.023, 0.0354, 0.0056, 0.0073, 0.0047, 0.0029, 0.0038, 0.0038, 0.0079, 0.0073, 0.0052, 0.0112, 0.0059, 0.0131, 0.0046, 0.0005, 0.0022, 0.006, 0.0374, 0.0442, 0.0082, 0.0018]

@MODELS.register_module()
class CrossEntropyLoss(BaseWeightedLoss):
    """Cross Entropy Loss.

    Support two kinds of labels and their corresponding loss type. It's worth
    mentioning that loss type will be detected by the shape of ``cls_score``
    and ``label``.
    1) Hard label: This label is an integer array and all of the elements are
        in the range [0, num_classes - 1]. This label's shape should be
        ``cls_score``'s shape with the `num_classes` dimension removed.
    2) Soft label(probability distribution over classes): This label is a
        probability distribution and all of the elements are in the range
        [0, 1]. This label's shape must be the same as ``cls_score``. For now,
        only 2-dim soft label is supported.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                CrossEntropy loss.

        Returns:
            torch.Tensor: The returned CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, 1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            # default reduction 'mean'
            if self.class_weight is not None:
                # Use weighted average as pytorch CrossEntropyLoss does.
                # For more information, please visit https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html # noqa
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label

            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)
            loss_cls = F.cross_entropy(cls_score, label, **kwargs)

        return loss_cls

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
    elif 48 <= x <= 51:
        return 6
    else:
        return None
'''
def hierarchical_label_smoothing(fine_labels, num_fine_classes, num_coarse_classes):
    smoothed_labels = torch.zeros((fine_labels.size(0), num_fine_classes))
    for i, fine_label in enumerate(fine_labels):
        coarse_label = fine2coarse(fine_label.item())
        for j in range(num_fine_classes):
            if fine2coarse(j) == coarse_label:
                smoothed_labels[i, j] = 0.1  # smoothing factor
        smoothed_labels[i, fine_label] = 0.9  # main label gets higher probability
    return smoothed_labels
'''

def fine2coarse_tensor():
    return torch.tensor([
        0, 0, 0, 0, 0,    # 0-4 -> 0
        1, 1, 1, 1, 1, 1, # 5-10 -> 1
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, # 11-23 -> 2
        3, 3, 3, 3, 3, 3, 3, 3, # 24-31 -> 3
        4, 4, 4, 4, 4, 4,  # 32-37 -> 4
        5, 5, 5, 5, 5, 5, 5, 5, 5, 5, # 38-47 -> 5
        6, 6, 6, 6 # 48-51 -> 6
    ])
# 0 anger 1 contempt 2 disgust 3 fear 4 happiness 5 sadness 6 surprise
# 0 negative 0       0         0      1 positive  0         2 surprise
def me_fine2coarse_tensor():
    return torch.tensor([
        0, 0, 0, 0, 1, 0, 2
    ])

def hierarchical_label_smoothing(fine_labels, num_fine_classes, num_coarse_classes):
    batch_size = fine_labels.size(0)
    
    # 初始化所有标签为均匀分布的平滑值
    smoothed_labels = torch.full((batch_size, num_fine_classes), 0.1 / (num_fine_classes - 1), device=fine_labels.device)
    
    # 获取细粒度标签对应的粗粒度标签
    fine2coarse_mapping = me_fine2coarse_tensor().to(fine_labels.device)
    coarse_labels = fine2coarse_mapping[fine_labels]
    
    # 创建掩码来识别属于相同粗粒度类别的细粒度标签
    fine2coarse_labels = fine2coarse_mapping.unsqueeze(0).expand(batch_size, -1)
    mask = (fine2coarse_labels == coarse_labels.unsqueeze(1))

    for i in range(batch_size):
        smoothed_labels[i, mask[i]] = 0.1 / mask[i].sum().float()
    
    # 将主标签的概率设为0.9
    smoothed_labels[torch.arange(batch_size), fine_labels] = 0.9

    return smoothed_labels

@MODELS.register_module()
class LabelSmoothingCrossEntropyLoss(BaseWeightedLoss):
    """Label Smoothing Cross Entropy Loss.

    Args:
        smoothing (float): The smoothing factor. Default to 0.1.
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Defaults to None.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 loss_weight: float = 1.0,
                 num_fine_classes: int = 7,
                 num_coarse_classes: int = 3,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.smoothing = smoothing
        self.loss_weight = loss_weight
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                Label Smoothing CrossEntropy loss.

        Returns:
            torch.Tensor: The returned Label Smoothing CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, dim=-1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(dim=-1)

            # default reduction 'mean'
            if self.class_weight is not None:
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                loss_cls = loss_cls.mean()
        else:
            # calculate loss for hard label
            smooth_label = hierarchical_label_smoothing(label, self.num_fine_classes, self.num_coarse_classes)
            #smooth_label = torch.full_like(cls_score, self.smoothing / (num_classes - 1))
            #smooth_label.scatter_(1, label.unsqueeze(1), 1 - self.smoothing)
            
            lsm = F.log_softmax(cls_score, dim=-1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(smooth_label * lsm).sum(dim=-1)

            # default reduction 'mean'
            if self.class_weight is not None:
                loss_cls = loss_cls.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * smooth_label)
            else:
                loss_cls = loss_cls.mean()
                
        return loss_cls

@MODELS.register_module()
class LabelSmoothingFocalLoss(BaseWeightedLoss):
    """Label Smoothing Cross Entropy Loss.

    Args:
        smoothing (float): The smoothing factor. Default to 0.1.
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Defaults to None.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 gamma: float = 2.0,
                 loss_weight: float = 1.0,
                 num_fine_classes: int = 7,
                 num_coarse_classes: int = 3,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.smoothing = smoothing
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                Label Smoothing CrossEntropy loss.

        Returns:
            torch.Tensor: The returned Label Smoothing CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, dim=1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            pt = torch.exp(-loss_cls)
            focal_loss = ((1 - pt) ** self.gamma) * loss_cls

            if self.class_weight is not None:
                focal_loss = focal_loss.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                focal_loss = focal_loss.mean()
        else:
            # calculate loss for hard label
            smooth_label = hierarchical_label_smoothing(label, self.num_fine_classes, self.num_coarse_classes)
            #smooth_label = torch.full_like(cls_score, self.smoothing / (num_classes - 1))
            #smooth_label.scatter_(1, label.unsqueeze(1), 1 - self.smoothing)
            
            lsm = F.log_softmax(cls_score, dim=-1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(smooth_label * lsm).sum(dim=-1)

            pt = torch.exp(-loss_cls)
            focal_loss = ((1 - pt) ** self.gamma) * loss_cls

            if self.class_weight is not None:
                focal_loss = focal_loss.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                focal_loss = focal_loss.mean()
                
        return focal_loss

@MODELS.register_module()
class LabelSmoothingCrossEntropyFocalLoss(BaseWeightedLoss):
    """Label Smoothing Cross Entropy Loss.

    Args:
        smoothing (float): The smoothing factor. Default to 0.1.
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Defaults to None.
    """

    def __init__(self,
                 smoothing: float = 0.1,
                 gamma: float = 2.0,
                 ce_loss_weight: float = 0.6,
                 loss_weight: float = 1.0,
                 num_fine_classes: int = 7,
                 num_coarse_classes: int = 3,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.smoothing = smoothing
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)
        self.num_fine_classes = num_fine_classes
        self.num_coarse_classes = num_coarse_classes
        self.ce_loss_weight = ce_loss_weight

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                Label Smoothing CrossEntropy loss.

        Returns:
            torch.Tensor: The returned Label Smoothing CrossEntropy loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label

            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, dim=1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)
            ce_loss = loss_cls.mean()

            pt = torch.exp(-loss_cls)
            focal_loss = ((1 - pt) ** self.gamma) * loss_cls

            if self.class_weight is not None:
                focal_loss = focal_loss.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                focal_loss = focal_loss.mean()
        else:
            # calculate loss for hard label
            smooth_label = hierarchical_label_smoothing(label, self.num_fine_classes, self.num_coarse_classes)
            #smooth_label = torch.full_like(cls_score, self.smoothing / (num_classes - 1))
            #smooth_label.scatter_(1, label.unsqueeze(1), 1 - self.smoothing)
            
            lsm = F.log_softmax(cls_score, dim=-1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(smooth_label * lsm).sum(dim=-1)

            ce_loss = loss_cls.mean()

            pt = torch.exp(-loss_cls)
            focal_loss = ((1 - pt) ** self.gamma) * loss_cls

            if self.class_weight is not None:
                focal_loss = focal_loss.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                focal_loss = focal_loss.mean()
                
        return (focal_loss * (1.0 - self.ce_loss_weight)) + (ce_loss * self.ce_loss_weight)


@MODELS.register_module()
class FocalLoss(BaseWeightedLoss):
    """Focal Loss.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        gamma (float): Focusing parameter. Defaults to 2.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 gamma: float = 2.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.gamma = gamma
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                Focal loss.

        Returns:
            torch.Tensor: The returned Focal loss.
        """
        if cls_score.size() == label.size():
            # calculate loss for soft label
            assert cls_score.dim() == 2, 'Only support 2-dim soft label'
            assert len(kwargs) == 0, \
                ('For now, no extra args are supported for soft label, '
                 f'but get {kwargs}')

            lsm = F.log_softmax(cls_score, dim=1)
            if self.class_weight is not None:
                self.class_weight = self.class_weight.to(cls_score.device)
                lsm = lsm * self.class_weight.unsqueeze(0)
            loss_cls = -(label * lsm).sum(1)

            pt = torch.exp(-loss_cls)
            focal_loss = ((1 - pt) ** self.gamma) * loss_cls

            if self.class_weight is not None:
                focal_loss = focal_loss.sum() / torch.sum(
                    self.class_weight.unsqueeze(0) * label)
            else:
                focal_loss = focal_loss.mean()
        else:
            # calculate loss for hard label
            if self.class_weight is not None:
                assert 'weight' not in kwargs, \
                    "The key 'weight' already exists."
                kwargs['weight'] = self.class_weight.to(cls_score.device)

            logpt = F.log_softmax(cls_score, dim=1)
            pt = torch.exp(logpt)
            logpt = logpt.gather(1, label.unsqueeze(1)).view(-1)
            pt = pt.gather(1, label.unsqueeze(1)).view(-1)
            focal_loss = -((1 - pt) ** self.gamma) * logpt

            if 'reduction' in kwargs and kwargs['reduction'] == 'sum':
                focal_loss = focal_loss.sum()
            else:
                focal_loss = focal_loss.mean()

        return focal_loss


@MODELS.register_module()
class BCELossWithLogits(BaseWeightedLoss):
    """Binary Cross Entropy Loss with logits.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        class_weight (list[float] | None): Loss weight for each class. If set
            as None, use the same weight 1 for all classes. Only applies
            to CrossEntropyLoss and BCELossWithLogits (should not be set when
            using other losses). Defaults to None.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 class_weight: Optional[List[float]] = None) -> None:
        super().__init__(loss_weight=loss_weight)
        self.class_weight = None
        if class_weight is not None:
            self.class_weight = torch.Tensor(class_weight)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        if self.class_weight is not None:
            assert 'weight' not in kwargs, "The key 'weight' already exists."
            kwargs['weight'] = self.class_weight.to(cls_score.device)
        loss_cls = F.binary_cross_entropy_with_logits(cls_score, label,
                                                      **kwargs)
        return loss_cls


@MODELS.register_module()
class CBFocalLoss(BaseWeightedLoss):
    """Class Balanced Focal Loss. Adapted from https://github.com/abhinanda-
    punnakkal/BABEL/. This loss is used in the skeleton-based action
    recognition baseline for BABEL.

    Args:
        loss_weight (float): Factor scalar multiplied on the loss.
            Defaults to 1.0.
        samples_per_cls (list[int]): The number of samples per class.
            Defaults to [].
        beta (float): Hyperparameter that controls the per class loss weight.
            Defaults to 0.9999.
        gamma (float): Hyperparameter of the focal loss. Defaults to 2.0.
    """

    def __init__(self,
                 loss_weight: float = 1.0,
                 samples_per_cls: List[int] = [309, 150, 20, 9, 156, 337, 242, 456, 962, 1283, 1193, 227, 112, 95, 42, 82, 76, 355, 87, 277, 147, 356, 373, 397, 238, 66, 92, 97, 60, 231, 259, 398, 63, 82, 53, 33, 43, 43, 89, 82, 58, 126, 66, 147, 52, 6, 25, 68, 421, 497, 92, 20],
                 beta: float = 0.9999,
                 gamma: float = 2.) -> None:
        super().__init__(loss_weight=loss_weight)
        self.samples_per_cls = samples_per_cls
        self.beta = beta
        self.gamma = gamma
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * len(weights)
        self.weights = weights
        self.num_classes = len(weights)

    def _forward(self, cls_score: torch.Tensor, label: torch.Tensor,
                 **kwargs) -> torch.Tensor:
        """Forward function.

        Args:
            cls_score (torch.Tensor): The class score.
            label (torch.Tensor): The ground truth label.
            kwargs: Any keyword argument to be used to calculate
                bce loss with logits.

        Returns:
            torch.Tensor: The returned bce loss with logits.
        """
        weights = torch.tensor(self.weights).float().to(cls_score.device)
        label_one_hot = F.one_hot(label, self.num_classes).float()
        weights = weights.unsqueeze(0)
        weights = weights.repeat(label_one_hot.shape[0], 1) * label_one_hot
        weights = weights.sum(1)
        weights = weights.unsqueeze(1)
        weights = weights.repeat(1, self.num_classes)

        BCELoss = F.binary_cross_entropy_with_logits(
            input=cls_score, target=label_one_hot, reduction='none')

        modulator = 1.0
        if self.gamma:
            modulator = torch.exp(-self.gamma * label_one_hot * cls_score -
                                  self.gamma *
                                  torch.log(1 + torch.exp(-1.0 * cls_score)))

        loss = modulator * BCELoss
        weighted_loss = weights * loss

        focal_loss = torch.sum(weighted_loss)
        focal_loss /= torch.sum(label_one_hot)

        return focal_loss
