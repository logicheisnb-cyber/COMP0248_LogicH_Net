#!/bin/bash
: <<'INFO'
This script contains the main commands for training and evaluating the two
implemented RGB-D models: `baseline` and `LogicH`.

Use this file if you want a quick start without reading the full README.
Before running it, make sure:
1. the Python environment is installed and activated
2. the dependencies in requirements.txt are installed
3. the dataset is available under the `dataset/` folder

The script will:
- train the baseline model
- train the LogicH model
- evaluate both models on the test split
- evaluate both models on the validation split
INFO

# ===========================================
# Training Commands
# ===========================================

# training the baseline model
python -m src.train --dataset_root dataset --model baseline --epochs 50 --batch_size 32

# training the LogicH model
python -m src.train --dataset_root dataset --model LogicH --epochs 50 --batch_size 32

# ===========================================
# Evaluation Commands
# ===========================================

# baseline evaluation
# dataset: dataset/test
python -m src.evaluate --split test --model baseline --ckpt weights/best_baseline.pt --test_root dataset/test --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json
# dataset: dataset/val
python -m src.evaluate --split val --model baseline --ckpt weights/best_baseline.pt --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json

# LogicH evaluation
# dataset: dataset/test
python -m src.evaluate --split test --model LogicH --ckpt weights/best_LogicH.pt --test_root dataset/test --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json
# dataset: dataset/val
python -m src.evaluate --split val --model LogicH --ckpt weights/best_LogicH.pt --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json


