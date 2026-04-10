# Copyright (c) OpenMMLab. All rights reserved.
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmengine.utils import is_list_of
#from mmengine.model.base_module import BaseModule
from mmengine.model import BaseModel
#from mmengine.model.base_model.data_preprocessor import BaseDataPreprocessor


class EpochBaseModel(BaseModel):
    """Base class for all algorithmic models.

    BaseModel implements the basic functions of the algorithmic model, such as
    weights initialize, batch inputs preprocess(see more information in
    :class:`BaseDataPreprocessor`), parse losses, and update model parameters.

    Subclasses inherit from BaseModel only need to implement the forward
    method, which implements the logic to calculate loss and predictions,
    then can be trained in the runner.

    Examples:
        >>> @MODELS.register_module()
        >>> class ToyModel(BaseModel):
        >>>
        >>>     def __init__(self):
        >>>         super().__init__()
        >>>         self.backbone = nn.Sequential()
        >>>         self.backbone.add_module('conv1', nn.Conv2d(3, 6, 5))
        >>>         self.backbone.add_module('pool', nn.MaxPool2d(2, 2))
        >>>         self.backbone.add_module('conv2', nn.Conv2d(6, 16, 5))
        >>>         self.backbone.add_module('fc1', nn.Linear(16 * 5 * 5, 120))
        >>>         self.backbone.add_module('fc2', nn.Linear(120, 84))
        >>>         self.backbone.add_module('fc3', nn.Linear(84, 10))
        >>>
        >>>         self.criterion = nn.CrossEntropyLoss()
        >>>
        >>>     def forward(self, batch_inputs, data_samples, mode='tensor'):
        >>>         data_samples = torch.stack(data_samples)
        >>>         if mode == 'tensor':
        >>>             return self.backbone(batch_inputs)
        >>>         elif mode == 'predict':
        >>>             feats = self.backbone(batch_inputs)
        >>>             predictions = torch.argmax(feats, 1)
        >>>             return predictions
        >>>         elif mode == 'loss':
        >>>             feats = self.backbone(batch_inputs)
        >>>             loss = self.criterion(feats, data_samples)
        >>>             return dict(loss=loss)

    Args:
        data_preprocessor (dict, optional): The pre-process config of
            :class:`BaseDataPreprocessor`.
        init_cfg (dict, optional): The weight initialized config for
            :class:`BaseModule`.

    Attributes:
        data_preprocessor (:obj:`BaseDataPreprocessor`): Used for
            pre-processing data sampled by dataloader to the format accepted by
            :meth:`forward`.
        init_cfg (dict, optional): Initialization config dict.
    """

    def __init__(self,
                 data_preprocessor: Optional[Union[dict, nn.Module]] = None,
                 init_cfg: Optional[dict] = None):
        super(EpochBaseModel, self).__init__(data_preprocessor, init_cfg)

    def train_step(self, epoch: int, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        """Implements the default model training process including
        preprocessing, model forward propagation, loss calculation,
        optimization, and back-propagation.

        During non-distributed training. If subclasses do not override the
        :meth:`train_step`, :class:`EpochBasedTrainLoop` or
        :class:`IterBasedTrainLoop` will call this method to update model
        parameters. The default parameter update process is as follows:

        1. Calls ``self.data_processor(data, training=False)`` to collect
           batch_inputs and corresponding data_samples(labels).
        2. Calls ``self(batch_inputs, data_samples, mode='loss')`` to get raw
           loss
        3. Calls ``self.parse_losses`` to get ``parsed_losses`` tensor used to
           backward and dict of loss tensor used to log messages.
        4. Calls ``optim_wrapper.update_params(loss)`` to update model.

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            optim_wrapper (OptimWrapper): OptimWrapper instance
                used to update model parameters.

        Returns:
            Dict[str, torch.Tensor]: A ``dict`` of tensor for logging.
        """
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(epoch, data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        return log_vars

    def val_step(self, data: Union[tuple, dict, list]) -> list:
        """Gets the predictions of given data.

        Calls ``self.data_preprocessor(data, False)`` and
        ``self(inputs, data_sample, mode='predict')`` in order. Return the
        predictions which will be passed to evaluator.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(0, data, mode='predict')  # type: ignore

    def test_step(self, data: Union[dict, tuple, list]) -> list:
        """``BaseModel`` implements ``test_step`` the same as ``val_step``.

        Args:
            data (dict or tuple or list): Data sampled from dataset.

        Returns:
            list: The predictions of given data.
        """
        data = self.data_preprocessor(data, False)
        return self._run_forward(0, data, mode='predict')  # type: ignore


    def _run_forward(self, epoch: int, data: Union[dict, tuple, list],
                     mode: str) -> Union[Dict[str, torch.Tensor], list]:
        """Unpacks data for :meth:`forward`

        Args:
            data (dict or tuple or list): Data sampled from dataset.
            mode (str): Mode of forward.

        Returns:
            dict or list: Results of training or testing mode.
        """
        if isinstance(data, dict):
            results = self(**data, epoch=epoch, mode=mode)
        elif isinstance(data, (list, tuple)):
            results = self(*data, epoch=epoch, mode=mode)
        else:
            raise TypeError('Output of `data_preprocessor` should be '
                            f'list, tuple or dict, but got {type(data)}')
        return results