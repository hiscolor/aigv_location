import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats
from ..utils import remove_duplicate_annotations


@register_dataset("lavdf_videomae")
class LAVDFVideoMAEDataset(Dataset):
    """
    LAV-DF dataset with VideoMAE features.

    Key differences from the TSN-based LAVDFDataset:
      - Features are single-stream (no rgb/flow split), stored as {feat_folder}/{video_id}.npy
      - file_prefix is ignored (no subdirectory structure per modality)
      - Supports configurable feat_folder / json_file so the same class works
        for different feature extractors as long as the naming convention matches.
    """
    def __init__(
        self,
        is_training,
        split,
        feat_folder,
        json_file,
        feat_stride,
        num_frames,
        default_fps,
        downsample_rate,
        max_seq_len,
        trunc_thresh,
        crop_ratio,
        input_dim,
        num_classes,
        file_ext,
        force_upsampling,
        # the following are accepted for interface compatibility but unused
        audio_feat_folder=None,
        audio_input_dim=0,
        file_prefix=None,
        audio_file_ext=None,
    ):
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio is None or len(crop_ratio) == 2

        self.feat_folder = feat_folder
        self.file_ext = file_ext
        self.json_file = json_file
        self.force_upsampling = force_upsampling

        self.split = split
        self.is_training = is_training

        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = {'Fake': 0}
        self.crop_ratio = crop_ratio

        dict_db = self._load_json_db(self.json_file)
        assert (num_classes == 1)
        self.data_list = dict_db

        self.db_attributes = {
            'dataset_name': 'LAVDF_VideoMAE',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            'empty_label_ids': []
        }
        print("{} subset has {} videos".format(self.split, len(self.data_list)))

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        with open(json_file, 'r') as fid:
            json_db = json.load(fid)

        dict_db = tuple()
        for value in json_db:
            key = os.path.splitext(os.path.basename(value['file']))[0]

            if value['split'].lower() not in self.split:
                continue

            feat_file = os.path.join(self.feat_folder, key + self.file_ext)
            if not os.path.exists(feat_file):
                continue

            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            elif 'video_frames' in value:
                fps = value['video_frames'] / value['duration']
            else:
                assert False, "Unknown video FPS."
            duration = value['duration']

            if ('fake_periods' in value) and (len(value['fake_periods']) > 0):
                valid_acts = value['fake_periods']
                num_acts = len(valid_acts)
                segments = np.zeros([num_acts, 2], dtype=np.float32)
                labels = np.zeros([num_acts, ], dtype=np.int64)
                for idx, act in enumerate(valid_acts):
                    segments[idx][0] = act[0]
                    segments[idx][1] = act[1]
                    labels[idx] = 0
            else:
                if self.is_training:
                    continue
                else:
                    segments = None
                    labels = None

            dict_db += ({
                'id': key,
                'fps': fps,
                'duration': duration,
                'segments': segments,
                'labels': labels
            }, )

        return dict_db

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self._get_item_safe(idx)

    def _get_item_safe(self, idx):
        video_item = self.data_list[idx]

        filename = os.path.join(
            self.feat_folder, video_item['id'] + self.file_ext)
        feats = np.load(filename).astype(np.float32)

        if self.feat_stride > 0 and (not self.force_upsampling):
            feat_stride, num_frames = self.feat_stride, self.num_frames
            if self.downsample_rate > 1:
                feats = feats[::self.downsample_rate, :]
                feat_stride = self.feat_stride * self.downsample_rate
        elif self.feat_stride > 0 and self.force_upsampling:
            feat_stride = float(
                (feats.shape[0] - 1) * self.feat_stride + self.num_frames
            ) / self.max_seq_len
            num_frames = feat_stride
        else:
            seq_len = feats.shape[0]
            assert seq_len <= self.max_seq_len
            if self.force_upsampling:
                seq_len = self.max_seq_len
            feat_stride = video_item['duration'] * video_item['fps'] / seq_len
            num_frames = feat_stride
        feat_offset = 0.5 * num_frames / feat_stride

        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        if (feats.shape[-1] != self.max_seq_len) and self.force_upsampling:
            resize_feats = F.interpolate(
                feats.unsqueeze(0),
                size=self.max_seq_len,
                mode='linear',
                align_corners=False
            )
            feats = resize_feats.squeeze(0)

        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                video_item['segments'] * video_item['fps'] / feat_stride
                - feat_offset
            )
            labels = torch.from_numpy(video_item['labels'])
            if self.is_training:
                vid_len = feats.shape[1] + feat_offset
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        continue
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        valid_label_list.append(label.view(1))
                if len(valid_seg_list) > 0:
                    segments = torch.stack(valid_seg_list, dim=0)
                    labels = torch.cat(valid_label_list)
                else:
                    segments = None
                    labels = None
        else:
            segments, labels = None, None

        data_dict = {
            'video_id': video_item['id'],
            'feats': feats,              # C x T
            'segments': segments,        # N x 2
            'labels': labels,            # N
            'fps': video_item['fps'],
            'duration': video_item['duration'],
            'feat_stride': feat_stride,
            'feat_num_frames': num_frames
        }

        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh,
                feat_offset, self.crop_ratio
            )

        if self.is_training and data_dict['segments'] is None:
            return self._get_item_safe((idx + 1) % len(self.data_list))

        return data_dict
