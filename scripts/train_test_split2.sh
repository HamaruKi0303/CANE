#!/bin/bash

python utils/cocosplit.py \
    --annotation-path     /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/downloaded-annotations/result.json \
    --split-ratio         0.85 \
    --train               /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/downloaded-annotations/train3.json \
    --test                /content/drive/MyDrive/PROJECT/201_HaMaruki/201_32_Layout_parser/layout-parser-main/downloaded-annotations/test3.json