#!/bin/bash

python /content/drive/MyDrive/PROJECT/201_HaMaruki/201.32_Layout_parser/layout-model-training-master/layout-model-training-master/tools/cocosplit.py \
    -s 0.8 \
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201.32_Layout_parser/layout-parser-main/datasets/_publaynet/labels/publaynet/val.json \
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201.32_Layout_parser/layout-parser-main/datasets/_publaynet/labels/publaynet/train2.json \
    /content/drive/MyDrive/PROJECT/201_HaMaruki/201.32_Layout_parser/layout-parser-main/datasets/_publaynet/labels/publaynet/test2.json