from typing import Any, Dict, List, Optional, Tuple, Union
import os
import mmengine
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from mmengine.dist import all_gather, get_rank
from mmengine.model import BaseModel, merge_dict
from mmengine.structures import LabelData

from mmaction.registry import MODELS
from .clip.clip_text import clip_text_l14, clip_text_b16


def visualize_video_batch(inputs, data_samples, output_dir="batch_videos/", num_samples=3, num_frames=16,
                            mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):
    os.makedirs(output_dir, exist_ok=True)
    # 转换 tensor 为 numpy 数组
    print(inputs.shape)
    inputs_np = inputs.cpu().numpy()

    # 获取 batch 中前 num_samples 个视频
    for sample_idx in range(num_samples):
        # 获取第 sample_idx 个视频的所有帧
        video = inputs_np[sample_idx]
        video_name = str(data_samples[sample_idx].filename)
        print(sample_idx)
        print(video_name)

        # 转换形状为 (T, H, W, C) 以便 matplotlib 可视化
        video = np.transpose(video, (1, 2, 3, 0))
        video = (video * std) + mean

        # 获取前 num_frames 帧进行可视化
        frames_to_show = min(num_frames, video.shape[0])
        
        for frame_idx in range(frames_to_show):
            print("frame: ", frame_idx)
            frame = video[frame_idx]
            minValue_raw = frame.min()
            frame = frame - minValue_raw
            maxValue_raw = frame.max()
            frame = frame*255 / maxValue_raw #normalize，将图像数据扩展到[0,255]
            frame = np.uint8(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(output_dir, f'sample_{sample_idx + 1}_frame_{frame_idx + 1}.png'), frame)
            '''
            # 创建图像
            plt.imshow(frame)
            plt.axis('off')

            # 保存图像到指定目录
            output_path = os.path.join(output_dir, f'sample_{sample_idx + 1}_frame_{frame_idx + 1}.png')
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
            '''


class GatherLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx: Any, input: torch.Tensor) -> Tuple[List]:
        ctx.save_for_backward(input)
        output = all_gather(input)
        return tuple(output)

    @staticmethod
    def backward(ctx: Any, *grads: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[get_rank()]
        return grad_out

def dynamic_weight(epoch, total_epochs, initial_weight=1.0, final_weight=0.1):
    """Adjusts the weight dynamically over epochs."""
    weight = initial_weight - (initial_weight - final_weight) * (epoch / total_epochs)
    return weight

def gaussian_dynamic_weight(epoch, total_epochs, initial_weight=1.0, final_weight=0.1):
    """Adjusts the weight dynamically using Gaussian decay over epochs."""
    #mean = total_epochs / 2  # Center of the Gaussian decay
    #sigma = total_epochs / 4  # Width of the Gaussian decay
    mean = 0
    sigma = total_epochs / 2
    #if epoch>total_epochs:
    #    epoch = total_epochs
    
    gaussian_decay = np.exp(-0.5 * ((epoch - mean) / sigma) ** 2)
    weight = final_weight + (initial_weight - final_weight) * gaussian_decay
    
    return weight

def gaussian_dynamic_weight_2(epoch, total_epochs, initial_weight=1.0, final_weight=0.1):
    """Adjusts the weight dynamically using Gaussian decay over epochs."""
    mean = total_epochs / 2  # Center of the Gaussian decay
    sigma = total_epochs / 4  # Width of the Gaussian decay
    
    gaussian_decay = np.exp(-0.5 * ((epoch - mean) / sigma) ** 2)
    weight = final_weight + (initial_weight - final_weight) * gaussian_decay
    
    return weight


@MODELS.register_module()
class ActionClip_MA_WithCls_ME(BaseModel):

    def __init__(self,
                 vision_encoder_config: Optional[Dict] = None,
                 text_encoder_config: Optional[Dict] = None,
                 text_encoder_pretrain: str = "resources/ViCLIP-B_InternVid-FLT-10M.pth",
                 freeze_text: bool = True,
                 to_float32: bool = False,
                 labels_or_label_file: Optional[Union[List[str], str]] = None,
                 templates_or_template_file: Optional[Union[List[str],
                                                            str]] = None,
                 data_preprocessor: Optional[Dict] = None,
                 loss: Dict = dict(type='CrossEntropyLoss', loss_weight=0.5),
                 cls_head_config: Optional[Dict] = None,
                 clip_loss_ratio: float = 1.0,
                 progressive: bool = False,
                 gaussian: int = 0,
                 total_weight: float=2.0,
                 final_weight: float = 0.1,
                 total_epochs: int = 55,
                 freeze_textproj: bool = True,
                 visual_batch_video: bool = False):
        print(data_preprocessor)
        super(ActionClip_MA_WithCls_ME, self).__init__(data_preprocessor=data_preprocessor)
    

        self.build_vision_encoder(vision_encoder_config)
        self.text_encoder_pretrain = text_encoder_pretrain
        self.build_text_encoder(text_encoder_config)
        self.context_length = text_encoder_config['context_length']

        self.loss = MODELS.build(loss)
        
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # Freeze weights
        self.freeze_textproj = freeze_textproj
        if freeze_text:
            self.freeze_text()
        
        if labels_or_label_file is not None:
            self.prompt, self.num_prompt = self.text_prompt(
                    labels_or_label_file, templates_or_template_file)

        if cls_head_config is not None:
            self.cls_head = MODELS.build(cls_head_config)
            print("cls_head build down! ", cls_head_config['type'])
        self.clip_loss_ratio = clip_loss_ratio
        if progressive:
            if gaussian == 1:
                self.progressive_weights = [gaussian_dynamic_weight(epoch, total_epochs, clip_loss_ratio, final_weight) for epoch in range(total_epochs)]
            elif gaussian == 2:
                self.progressive_weights = [gaussian_dynamic_weight_2(epoch, total_epochs, clip_loss_ratio, final_weight) for epoch in range(total_epochs)]
            else:
                self.progressive_weights = [dynamic_weight(epoch, total_epochs, clip_loss_ratio, final_weight) for epoch in range(total_epochs)]
        self.progressive = progressive
        self.total_weight = total_weight
        self.visual_batch_video = visual_batch_video
        


    def freeze_text(self):
        """freeze text encoder"""
        if self.freeze_textproj:
            for p in self.text_encoder.parameters():
                p.requires_grad = False
        else:
            for name, param in self.text_encoder.named_parameters():
                if "text_projection" not in name:
                    if ".ln_final" not in name:
                        param.requires_grad = False
                    else:
                        param.requires_grad = True
                else:
                    param.requires_grad = True
            
        

    def no_weight_decay(self):
        ret = {"logit_scale"}
        ret.update(
            {"vision_encoder." + k for k in self.vision_encoder.no_weight_decay()}
        )
        ret.update(
            {"text_encoder." + k for k in self.text_encoder.no_weight_decay()}
        )
        ret.update(
            {"cls_head." + k for k in self.cls_head.no_weight_decay()}
        )

        return ret


    def build_vision_encoder(self, vision_encoder_config):
        """build vision encoder
        Returns: (vision_encoder, vision_layernorm). Each is a `nn.Module`.

        """
        encoder_name = vision_encoder_config['type']
        # if encoder_name != "vit_l14":
        #     raise ValueError(f"Not implemented: {encoder_name}")

        self.vision_encoder = MODELS.build(vision_encoder_config)
        print("Vision Encoder is ", encoder_name)
        

    def build_text_encoder(self, text_encoder_config):
        """build text_encoder and possiblly video-to-text multimodal fusion encoder.
        Returns: nn.Module. The text encoder

        """
        encoder_name = text_encoder_config['type']
        print(f'Text Encoder name: {encoder_name}')
        # if encoder_name != "vit_l14":
        #     raise ValueError(f"Not implemented: {encoder_name}")
        if encoder_name == "vit_b16":
            self.text_encoder = clip_text_b16(
                embed_dim=text_encoder_config['output_dim'],
                context_length=text_encoder_config['context_length'],
                pretrained=self.text_encoder_pretrain
            )
        else:
            raise ValueError(f"Not implemented: {encoder_name}")


    def get_text_encoder(self):
        """get text encoder, used for text and cross-modal encoding"""
        encoder = self.text_encoder
        return encoder.bert if hasattr(encoder, "bert") else encoder
    
    def text_prompt(self, labels_or_label_file, templates_or_template_file=None):
        if isinstance(labels_or_label_file, str):
            labels = mmengine.list_from_file(labels_or_label_file)
        elif isinstance(labels_or_label_file, list):
            labels = labels_or_label_file
        else:
            raise ValueError(f'`labels_or_label_file` must be `list` or `str`, '
                         f'but got {type(labels_or_label_file)}')
        print("Number of labels: " + str(len(labels)) + "\n")
        if templates_or_template_file is None:
            templates_simple = ['This micro-expression involves {}.']
            templates = [
            "This micro-expression involves {}.", "Key features of this micro-expression are {}.",
            "This micro-expression is characterized by {}.", "One can identify this micro-expression by {}.",
            "This micro-expression typically manifests as {}.", "The hallmark of this micro-expression is {}.",
            "An identifiable trait of this micro-expression is {}."
            ]
            templates_2 = [
            "A man wearing glasses is showing a micro-expression involving {}.",
            "This micro-expression involves {}.", "Key features of this micro-expression are {}.",
            "A woman wearing glasses is making a micro-expression by {}.",
            "One can identify this micro-expression by {}.",
            "An identifiable trait of this micro-expression is {}.",
            "A person is exhibiting a micro-expression characterized by {}.",
            ]
            templates_samm = [
            "A man with high nose bridge is showing a micro-expression involving {}.",
            "A man with deep-set eyes is showing a micro-expression characterized by {}.",
            "A person with high nose bridge and deep-set eyes is making a micro-expression by {}.",
            "This micro-expression typically manifests as {}.", "Key features of this micro-expression are {}.",
            "An identifiable trait of this micro-expression is {}."
            ]
        elif isinstance(templates_or_template_file, str):
            templates = mmengine.list_from_file(templates_or_template_file)
        elif not mmengine.is_seq_of(templates_or_template_file, str):
            raise ValueError(f'`template` must be list of `str`, `str` or `None`, '
                         f'but got {type(templates_or_template_file)}')

        num_prompt = len(templates)
        prompt = torch.cat(
            [self.text_encoder.tokenize(t.format(c), context_length=self.context_length) for t in templates for c in labels])
        return prompt, num_prompt
    

    def forward(self, epoch: int=0,
                inputs: torch.Tensor=None,
                data_samples: Optional[List] = None,
                mode: str = 'tensor'):
        # Read epoch from model attribute if available (set by WithEpochBasedTrainLoop)
        if hasattr(self, '_current_epoch'):
            epoch = self._current_epoch

        num_segs = inputs.shape[1]
        # [N, num_crops, C, T, H, W] ->
        # [N * num_crops, C, T, H, W]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        inputs = inputs.view((-1, ) + inputs.shape[2:])
        if self.visual_batch_video and mode=='loss':
            visualize_video_batch(inputs, data_samples, output_dir="batch_videos/", num_samples=8, num_frames=16)
            self.visual_batch_video = False

        if mode == 'tensor':
            video_features, raw_video_features = self.vision_encoder(inputs)
            predict_kwargs = dict()
            cls_scores, pred_feat = self.cls_head(raw_video_features, **predict_kwargs)
            print("pred_feat: ", pred_feat.size())
            return pred_feat#raw_video_features, 

        elif mode == 'predict':
            assert hasattr(self, 'prompt'),\
                '`labels_or_label_file` is required to perform prediction. '

            video_features, raw_video_features = self.vision_encoder(inputs)
            '''
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)

            bsz = len(data_samples)
            num_views = video_features.shape[0] // bsz

            text_features = self.text_encoder(self.prompt.to(inputs.device))
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)

            # (bsz*num_views, num_prompt, num_classes) ->
            # (bsz, num_views*num_prompt, num_classes)
            similarity = (100.0 * video_features @ text_features.T). \
                view(bsz, num_views * self.num_prompt, -1)

            clip_cls_scores = F.softmax(similarity, dim=2).mean(dim=1)
            '''
            predict_kwargs = dict()
            data_samples, pred_feat = self.cls_head.predict(raw_video_features, data_samples, **predict_kwargs)
            '''
            for data_sample, score in zip(data_samples, clip_cls_scores):
                data_sample.set_pred_score(score)
                #pred_score = LabelData(item=score)
            '''
            return data_samples

        elif mode == 'loss': 

            video_features, raw_video_features = self.vision_encoder(inputs)
            #---------------clip-video
            video_features = video_features / video_features.norm(
                dim=-1, keepdim=True)
            #---------------clip-text
            text_id = np.random.randint(
                self.num_prompt, size=len(data_samples))
            #print(data_samples[0])
            real_au_labels = [x.au_label for x in data_samples]
            selected_prompt = self.prompt.view(
                self.num_prompt, -1,
                self.prompt.shape[-1])[text_id, real_au_labels].to(inputs.device)

            text_features = self.text_encoder(selected_prompt)
            text_features = text_features / text_features.norm(
                dim=-1, keepdim=True)
            #---------------clip-loss
            video_features = torch.cat(
                GatherLayer.apply(video_features), dim=0)
            text_features = torch.cat(GatherLayer.apply(text_features), dim=0)

            logit_scale = self.logit_scale.exp()
            logits_per_video = logit_scale * video_features @ text_features.t()
            logits_per_text = logits_per_video.t()
            labels = torch.arange(logits_per_video.shape[0]).to(
                logits_per_video.device)
            
            sim_loss_v2t = self.loss(logits_per_video, labels)
            sim_loss_t2v = self.loss(logits_per_text, labels)

            #---------------cls_loss
            loss_kwargs = dict()
            loss_cls_dict = self.cls_head.loss(raw_video_features, data_samples, **loss_kwargs)

            losses = dict()

            if self.progressive:
                try:
                    self.clip_loss_ratio = self.progressive_weights[epoch]
                except:
                    self.clip_loss_ratio = self.progressive_weights[-1]
                if self.total_weight>0:
                    loss_cls_dict['loss_cls'] = loss_cls_dict['loss_cls'] * (self.total_weight-self.clip_loss_ratio)
                    loss_cls_dict['loss_cls_weight'] = torch.tensor(self.total_weight-self.clip_loss_ratio).to(inputs.device)
            
            losses['sim_loss_v2t'] = sim_loss_v2t * self.clip_loss_ratio
            losses['sim_loss_t2v'] = sim_loss_t2v * self.clip_loss_ratio
            losses['sim_loss_weight'] = torch.tensor(self.clip_loss_ratio).to(inputs.device)
            #losses['loss_cls'] = loss_cls
            losses = merge_dict(loss_cls_dict, losses)
            #print(losses)
            return losses

        else:
            raise RuntimeError(
                f'Invalid mode "{mode}". '
                'Only supports `predict`, `loss` and `tensor` mode. ')
