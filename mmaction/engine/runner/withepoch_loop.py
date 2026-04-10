# Copyright (c) OpenMMLab. All rights reserved.
import gc
from typing import Dict, List, Union, Sequence

from mmengine.runner import EpochBasedTrainLoop
from torch.utils.data import DataLoader

from mmaction.registry import LOOPS


@LOOPS.register_module()
class WithEpochBasedTrainLoop(EpochBasedTrainLoop):
    """EpochBasedTrainLoop with multiple dataloaders.

    Args:
        runner (Runner): A reference of runner.
        dataloader (Dataloader or Dict): A dataloader object or a dict to
            build a dataloader for training the model.
        other_loaders (List of Dataloader or Dict): A list of other loaders.
            Each item in the list is a dataloader object or a dict to build
            a dataloader.
        max_epochs (int): Total training epochs.
        val_begin (int): The epoch that begins validating. Defaults to 1.
        val_interval (int): Validation interval. Defaults to 1.
    """

    def __init__(self,
                 runner,
                 dataloader: Union[Dict, DataLoader],
                 max_epochs: int,
                 val_begin: int = 1,
                 val_interval: int = 1) -> None:
        super(WithEpochBasedTrainLoop, self).__init__(runner, dataloader, max_epochs, val_begin,
                         val_interval)
        

    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one min-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_train_iter', batch_idx=idx, data_batch=data_batch)
        # Enable gradient accumulation mode and avoid unnecessary gradient
        # synchronization during gradient accumulation process.
        # outputs should be a dict of loss.
        # Set current epoch on the model so forward() can access it
        model = self.runner.model
        if hasattr(model, 'module'):
            model.module._current_epoch = self._epoch
        else:
            model._current_epoch = self._epoch
        outputs = self.runner.model.train_step(
            data=data_batch, optim_wrapper=self.runner.optim_wrapper)

        self.runner.call_hook(
            'after_train_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
        self._iter += 1