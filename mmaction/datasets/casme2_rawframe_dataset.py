# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from typing import Callable, List, Optional, Union

from mmengine.fileio import exists, list_from_file

from mmaction.registry import DATASETS
from mmaction.utils import ConfigType
from .base import BaseActionDataset


@DATASETS.register_module()
class CASME2RawFrameWithAUDataset(BaseActionDataset):
    """Video dataset for action recognition.

    The dataset loads raw videos and apply specified transforms to return a
    dict containing the frame tensors and other information.

    The ann_file is a text file with multiple lines, and each line indicates
    a sample video with the filepath and label, which are split with a
    whitespace. Example of a annotation file:

    .. code-block:: txt

        some/path/000.mp4 1
        some/path/001.mp4 1
        some/path/002.mp4 2
        some/path/003.mp4 2
        some/path/004.mp4 3
        some/path/005.mp4 3


    Args:
        ann_file (str): Path to the annotation file.
        pipeline (List[Union[dict, ConfigDict, Callable]]): A sequence of
            data transforms.
        data_prefix (dict or ConfigDict): Path to a directory where videos
            are held. Defaults to ``dict(video='')``.
        multi_class (bool): Determines whether the dataset is a multi-class
            dataset. Defaults to False.
        num_classes (int, optional): Number of classes of the dataset, used in
            multi-class datasets. Defaults to None.
        start_index (int): Specify a start index for frames in consideration of
            different filename format. However, when taking videos as input,
            it should be set to 0, since frames loaded from videos count
            from 0. Defaults to 0.
        modality (str): Modality of data. Support ``'RGB'``, ``'Flow'``.
            Defaults to ``'RGB'``.
        test_mode (bool): Store True when building test or validation dataset.
            Defaults to False.
        delimiter (str): Delimiter for the annotation file.
            Defaults to ``' '`` (whitespace).
    """

    def __init__(self,
                 emo_ann_file: str,
                 au_ann_file: str,
                 pipeline: List[Union[dict, Callable]],
                 data_prefix: ConfigType = dict(video=''),
                 subset: List = None,
                 multi_class: bool = False,
                 num_classes: Optional[int] = None,
                 start_index: int = 0,
                 modality: str = 'RGB',
                 test_mode: bool = False,
                 delimiter: str = ' ',
                 **kwargs) -> None:
        self.delimiter = delimiter
        self.au_ann_file = au_ann_file
        self.emo_ann_file = emo_ann_file
        self.subset = subset
        self.labels = []
        super().__init__(
            ann_file=emo_ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            multi_class=multi_class,
            num_classes=num_classes,
            start_index=start_index,
            modality=modality,
            test_mode=test_mode,
            **kwargs)
        

    def load_data_list(self) -> List[dict]:
        """Load annotation file to get video information."""
        exists(self.emo_ann_file)
        exists(self.au_ann_file)
        data_list = []
        emo_fin = list_from_file(self.emo_ann_file)
        au_fin = list_from_file(self.au_ann_file)
        assert len(emo_fin) == len(au_fin)
        for i in range(len(emo_fin)):
            line = emo_fin[i]
            au_line = au_fin[i]
            line_split = line.strip().split(self.delimiter)
            au_line_split = au_line.strip().split(self.delimiter)
            if self.multi_class:
                assert self.num_classes is not None
                filename, total_frames, label = line_split[0], line_split[1], line_split[2:]
                label = list(map(int, label))
            # add fake label for inference datalist without label
            elif len(line_split) == 1:
                filename, total_frames, label = line_split[0], -1, -1
            else:
                filename, total_frames, label = line_split
                au_filename, au_label = au_line_split
                assert filename == au_filename
                label = int(label)
                au_label = int(au_label)
                total_frames = int(total_frames)
            sub = int(filename.split("CA")[-1].split("_")[0])
            if sub in self.subset:
                if self.data_prefix['video'] is not None:
                    filename = osp.join(self.data_prefix['video'], filename)
                data_list.append(dict(filename=filename, total_frames=total_frames, label=label, au_label=au_label))
                self.labels.append(label)
        print("Total data number: ", len(data_list))
        return data_list
