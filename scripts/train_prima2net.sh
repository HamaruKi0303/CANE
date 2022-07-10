#!/bin/bash

python tools/train_net2.py \
    --dataset_name          prima-layout \
    --json_annotation_train /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/datasets/PRImADataset/annotations-train.json \
    --image_path_train      /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/datasets/PRImADataset/Images \
    --json_annotation_val   /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/datasets/PRImADataset/annotations-val.json \
    --image_path_val        /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/datasets/PRImADataset/Images \
    --config-file           configs/prima/mask_rcnn_R_50_FPN_3x.yaml \
    OUTPUT_DIR  output/PRImA/mask_rcnn_R_50_FPN_3x/001 \
    SOLVER.IMS_PER_BATCH 2 