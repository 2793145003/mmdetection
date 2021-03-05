import os
import json
import argparse
import numpy as np
import pandas as pd

import mmcv
from mmcv import Config

from mmdet.apis import train_detector
from mmdet.apis import set_random_seed
from mmdet.models import build_detector
from mmdet.datasets import build_dataset
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets.builder import DATASETS
from mmdet.utils.get_config_file import get_config_file_from_params

@DATASETS.register_module()
class OpenbayesDataset(CustomDataset):
    
    def __init__(self, **params):
        print("*"*50,params)
        self.CLASSES = params['classes']
        self.input_path = params['args']['input']
        self.image_width = params['image_width']
        self.image_height = params['image_height']
        super(OpenbayesDataset, self).__init__(
            ann_file=params['ann_file'],
            pipeline=params['pipeline'])

    def load_annotations(self, ann_file='train_meta.csv'):
        cat2label = {k: i for i, k in enumerate(self.CLASSES)}
        # load image list from file    
        data_infos = []
        meta_csv = pd.read_csv(os.path.join(self.input_path, ann_file))
        for i in range(len(meta_csv)):
            filename = meta_csv["json_Label"][i]
            json_file = os.path.join(self.input_path, filename)
            with open(json_file) as f:
                imgs_anns = json.load(f)
                filename = os.path.join(self.input_path, meta_csv["image_Source"][i])
                image = mmcv.imread(filename)
                height, width = image.shape[:2]
                # assert (height == self.image_height) and (width == self.image_width), 'Image size ({}, {}) should equal to the size in config ({}, {}).'.format(width, height, self.image_width, self.image_height)

                data_info = dict(filename=filename, width=width, height=height)
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                for bbox in imgs_anns["bboxes"]:
                    bbox_name = bbox["label"] 

                    if bbox_name in cat2label:
                        gt_labels.append(cat2label[bbox_name])
                        if "x_min" in bbox:
                            gt_bboxes.append([bbox["x_min"]*width, bbox["y_min"]*height, bbox["x_max"]*width, bbox["y_max"]*height])
                        else:
                            gt_bboxes.append([
                                min(bbox["x_arr"])*width,
                                min(bbox["y_arr"])*height,
                                max(bbox["x_arr"])*width,
                                max(bbox["y_arr"])*height,
                            ])
                    else:
                        print(bbox_name, cat2label)
                        gt_labels_ignore.append(-1)
                        gt_bboxes_ignore.append(bbox)

                data_anno = dict(
                    bboxes=np.array(gt_bboxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.array(gt_labels, dtype=np.long),
                    bboxes_ignore=np.array(gt_bboxes_ignore,
                                           dtype=np.float32).reshape(-1, 4),
                    labels_ignore=np.array(gt_labels_ignore, dtype=np.long))

                data_info.update(ann=data_anno)
                data_infos.append(data_info)

        return data_infos

def main():
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument('--input', '-i', help='input dataset path', default="/input0/")
    parser.add_argument('--output', '-o', help='output model path', default="/output/model_output")
    parser.add_argument('--model', '-m', help='algorithm', default="faster_rcnn")
    parser.add_argument('--hparams', '-p', help='hyper params json path', default="/output/mmdetection/openbayes_params.json")
    args, unknown = parser.parse_known_args()
    print(args)

    params = json.load(open(args.hparams))
    params['args'] = vars(args)
    # print(params)

    # default_parameters
    cfg = Config.fromfile(get_config_file_from_params(params))
    cfg.dataset_type = 'OpenbayesDataset'
    cfg.data.test.type = 'OpenbayesDataset'
    cfg.data.test.ann_file = 'train_meta.csv'
    cfg.data.train.type = 'OpenbayesDataset'
    cfg.data.train.ann_file = 'train_meta.csv'
    cfg.data.val.type = 'OpenbayesDataset'
    cfg.data.val.ann_file = 'val_meta.csv'
    cfg.model.roi_head.bbox_head.num_classes = len(params['classes'])
    # cfg.load_from = 'checkpoints/mask_rcnn_r50_caffe_fpn_mstrain-poly_3x_coco_bbox_mAP-0.408__segm_mAP-0.37_20200504_163245-42aa3d00.pth'
    cfg.work_dir = params['args']['output']
    cfg.evaluation.metric = params['metrics']
    cfg.evaluation.save_best = 'mAP'
    cfg.evaluation.rule = 'greater'

    cfg.total_epochs = params['epochs']
    cfg.data.samples_per_gpu = params['batch_size']

    cfg.optimizer.lr = 0.001
    cfg.lr_config.warmup = None
    cfg.log_config.interval = 10

    # seed
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    datasets = [build_dataset(cfg.data.train, params)]
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    model.CLASSES = datasets[0].CLASSES

    cfg.dump('openbayes_config.py')

    mmcv.mkdir_or_exist(os.path.abspath(cfg.work_dir))
    train_detector(model, datasets, cfg, distributed=False, validate=True, params=params)


if __name__ == '__main__':
    main()