# LogicH: RGB-D Hand Gesture Multi-Task Learning

This repository contains an RGB-D hand gesture recognition project built around multi-task learning. The model jointly performs:

- hand segmentation
- hand bounding box regression
- gesture classification

Two RGB-D models are implemented:

- `baseline`: a lightweight shared RGB-D encoder with standard fusion
- `LogicH`: an enhanced variant that adds classification-focused cross-modal fusion and depth-guided bounding box refinement

The codebase includes training, validation, test evaluation, and qualitative result visualization utilities.

## Project Scope

The system is designed for hand gesture understanding from paired RGB and depth inputs. Instead of training separate models for segmentation, localization, and recognition, the project learns the three tasks jointly in a single network.

The implementation is structured around:

- a shared RGB-D encoder
- a decoder for segmentation
- segmentation-derived hand localization
- gesture classification using masked pooled features
- evaluation scripts for validation and nested test folder structures

This README reflects the current code in the repository. It intentionally does not document `ini_dataset` or `model_hist`, since those folders are not part of the final deliverable workflow.

## Repository Structure

```text
.
|-- README.md
|-- requirements.txt
|-- run.sh
|-- dataset/
|   |-- rgb/
|   |-- annotation/
|   |-- depth/
|   `-- test/
`-- src/
    |-- dataset_convert.py
    |-- dataloader.py
    |-- evaluate.py
    |-- model.py
    |-- train.py
    |-- utils.py
    `-- visualise.py
```

## Implemented Models

### 1. Baseline

The baseline model uses:

- a lightweight RGB branch
- a lightweight depth branch
- FiLM-style depth modulation in early stages
- simple RGB-depth fusion blocks
- a decoder for segmentation
- segmentation-derived bounding box prediction
- multi-scale masked feature pooling for gesture classification

In this design, the bounding box is produced directly from the predicted segmentation mask using a soft mask-to-box conversion.

### 2. LogicH

`LogicH` keeps the same overall multi-task framework but changes two parts:

- `GatedCrossModalFusion` is used only for the classification pathway
- `DepthGuidedBoxRefiner` improves the initial box estimated from the segmentation mask

The intended effect is:

- stronger gesture recognition by refining RGB features with depth-aware gating
- more stable hand localization by using depth information inside a segmentation-derived ROI

## Model Architecture Summary

Both models share the same high-level pipeline:

1. RGB and depth inputs are encoded jointly.
2. A decoder predicts a binary hand mask.
3. A normalized hand bounding box is estimated.
4. Gesture classification is performed using masked pooled features from multiple scales.

### Shared Encoder

The encoder in `src/model.py` is `LiteRGBDEncoder`. It contains:

- depthwise separable stem blocks
- inverted residual blocks
- early depth-conditioned FiLM modulation
- stage-wise RGB-depth fusion

The encoder returns multi-scale feature maps:

- `s0`: 1/2 resolution
- `s1`: 1/4 resolution
- `s2`: 1/8 resolution
- `s3`: 1/16 resolution
- `s4`: 1/32 resolution

For `LogicH`, additional depth features from higher stages are also returned for classification refinement and box refinement.

### Segmentation Head

Both models use a U-Net-like upsampling path:

- `up3`
- `up2`
- `up1`
- `up0`

The final segmentation head predicts a single-channel hand mask logit map, which is then resized back to the input image size.

### Bounding Box Prediction

For the baseline:

- the predicted segmentation probability map is converted into a normalized box `[cx, cy, w, h]`

For `LogicH`:

- the segmentation mask first produces an initial box
- the box is expanded into an ROI
- pooled RGB and depth features inside that ROI are passed through an MLP refiner
- the predicted correction is added to the initial box

### Classification Head

Gesture classification uses segmentation-guided masked average pooling. Features are pooled from multiple scales and then passed to an MLP classifier.

The default number of classes is `10`.

## Gesture Classes

The class names defined in `src/dataloader.py` are:

```text
call, dislike, like, ok, one, palm, peace, rock, stop, three
```

These labels are used during training, validation, confusion matrix generation, and overlay visualization.

## Dataset Format

If your dataset is not already in the flat training format, read the conversion section below first before trying to train the model.

### Converting Nested Datasets to Training Format

Some input datasets may not initially follow the flat training structure used by `src/train.py`. In particular, you may receive data in a nested layout such as:

```text
dataset/
`-- 25045993_He/
    |-- G01_call/
    |   `-- clip01/
    |       |-- annotation/
    |       |-- depth/
    |       `-- rgb/
    `-- ...
```

This repository includes a conversion utility at `src/dataset_convert.py` to transform that nested clip-style structure into the flat training format:

```text
dataset/
|-- 25045993_He/              <- original nested input is kept
|-- annotation/G01/0001.png
|-- depth/G01/0001.png
`-- rgb/G01/0001.png
```

The script only copies frames that exist in all three folders:

- `annotation`
- `depth`
- `rgb`

Non-image files such as `.json` and `.npy` are ignored.

The conversion script can only be used when the input dataset follows this pattern:

- the input root points either directly to the gesture folders, or to a parent folder that contains exactly one such submission folder
- gesture folders must start with a gesture prefix such as `G01`, `G02`, ..., `G10`
- clip folders inside each gesture folder must be named like `clip01`, `clip02`, ...
- each clip folder must contain all three image folders:
  - `annotation`
  - `depth`
  - `rgb`
- frame files must be image files with matching stems across the three folders

Examples of valid gesture folder names:

- `G01_call`
- `G02_dislike`
- `G10_three`
- `G01`

Examples of valid clip folder names:

- `clip01`
- `clip02`
- `clip15`

The script is not intended for:

- already-flat training datasets in `annotation/depth/rgb/G0X` format
- datasets missing one or more required modality folders
- arbitrary folder names that do not start with a `Gxx` gesture prefix
- clip folders that do not start with `clip`

Before running the script, set the input and output paths near the top of `src/dataset_convert.py`:

```python
INPUT_ROOT = Path("dataset/25045993_He")
OUTPUT_ROOT = Path("dataset")
```

This is the recommended setup when:

- `dataset/25045993_He` contains the original nested clip-style dataset
- `dataset/` is where you want the generated `annotation/`, `depth/`, and `rgb/` folders to appear

Then run:

```bash
python -m src.dataset_convert
```

Important behavior:

- the original nested input folder is kept
- only `dataset/annotation`, `dataset/depth`, and `dataset/rgb` are rebuilt
- output file names are renumbered per gesture using four digits, for example `0001.png`
- gesture IDs are detected from the prefix only, such as `G01`, `G02`, ..., `G10`

If your dataset is already in flat training format, the script will detect that and do nothing.

### Training / Validation Dataset

The training loader expects this structure:

```text
dataset/
|-- rgb/
|   |-- G01/
|   |-- G02/
|   `-- ...
|-- annotation/
|   |-- G01/
|   |-- G02/
|   `-- ...
`-- depth/
    |-- G01/
    |-- G02/
    `-- ...
```

Each gesture folder should contain matching file names across:

- `rgb`
- `annotation`
- `depth`

Example:

```text
dataset/
|-- rgb/G01/0001.png
|-- annotation/G01/0001.png
`-- depth/G01/0001.png
```

### Test Dataset

The test loader expects a nested structure:

```text
dataset/test/
|-- G01_call/
|   |-- clip01/
|   |   |-- rgb/
|   |   |-- annotation/
|   |   `-- depth/
|   `-- clip02/
|-- G02_dislike/
`-- ...
```

Example:

```text
dataset/test/G01_call/clip01/rgb/frame_001.png
dataset/test/G01_call/clip01/annotation/frame_001.png
dataset/test/G01_call/clip01/depth/frame_001.png
```

The evaluation script infers the gesture label from the top-level folder name. It supports patterns such as:

- `G01_call`
- `call`
- `G01`

## Data Processing

### Input Modalities

The current implementation is RGB-D only. There is no separate RGB-only model in the current codebase.

- RGB input shape: `(3, H, W)`
- depth input shape: `(1, H, W)`
- mask target shape: `(1, H, W)`
- box target format: normalized `(cx, cy, w, h)`

### Default Image Size

The command-line default for training and evaluation is:

```text
256 x 256
```

The dataset class itself contains an internal default of `(240, 320)`, but the training and evaluation scripts explicitly pass `--image_size` defaults of `256 256`, so in normal CLI usage the effective default is `256 x 256`.

### Training Augmentations

The training dataset applies:

- random horizontal flip
- random affine transform
- RGB-only color jitter
- optional RGB-only Gaussian blur

The affine transform includes:

- rotation
- translation
- scaling

Masks use nearest-neighbor interpolation to preserve binary structure. Depth and RGB images use bilinear interpolation.

## Installation

### 1. Create an Environment

Using `venv`:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If you use Conda:

```bash
conda create -n logich python=3.10
conda activate logich
```

### 2. Install Dependencies

If `requirements.txt` is populated in your local copy:

```bash
pip install -r requirements.txt
```

At minimum, the code requires packages from the following stack:

- `torch`
- `torchvision`
- `numpy`
- `Pillow`
- `matplotlib`

## Training

Training is launched through `src/train.py`.

### Available Models

- `baseline`
- `LogicH`

Both values are accepted from the CLI. Model name matching is normalized internally, so `LogicH` is the intended command-line form.

### Default Training Command

Baseline:

```bash
python -m src.train --dataset_root dataset --model baseline --epochs 50 --batch_size 32
```

LogicH:

```bash
python -m src.train --dataset_root dataset --model LogicH --epochs 50 --batch_size 32
```

### Important Training Arguments

```text
--dataset_root   root folder of the training dataset
--model          baseline or LogicH
--epochs         number of training epochs
--batch_size     batch size
--lr             learning rate
--weight_decay   AdamW weight decay
--num_workers    DataLoader workers
--image_size     input size, passed as H W
--val_ratio      validation split ratio
--seed           random seed
--save_dir       checkpoint output directory
--num_classes    number of gesture classes
--width          base channel width for the encoder
--min_lr         minimum LR for cosine annealing
--lambda_seg     segmentation loss weight
--lambda_box     box regression loss weight
--lambda_cls     classification loss weight
```

### Training Objective

The total training loss is:

```text
total_loss = lambda_seg * segmentation_loss
           + lambda_box * box_loss
           + lambda_cls * classification_loss
```

where:

- segmentation loss = `BCEWithLogitsLoss + DiceLoss`
- box loss = `SmoothL1Loss`
- classification loss = `CrossEntropyLoss`

### Validation Metrics

At the end of each epoch, the validation stage reports:

- `mIoU`
- `Dice`
- `BoxIoU`
- `DetAcc@0.5`
- `ClsAcc`
- `MacroF1`

The best checkpoint is selected using validation `mIoU`.

### Saved Checkpoints

Checkpoints are saved under `weights/` by default:

```text
weights/best_baseline.pt
weights/best_LogicH.pt
```

Each checkpoint contains:

- epoch
- model name
- model state dict
- optimizer state dict
- best validation mIoU
- CLI arguments
- summary validation metrics

## Evaluation

Evaluation is implemented in `src/evaluate.py`.

Two split types are supported:

- `val`
- `test`

### Validation Evaluation

Validation evaluation rebuilds the validation split from `dataset_root` using:

- the same seed
- the same validation ratio

This is important: if you want the evaluated validation set to match training-time validation exactly, keep `--seed` and `--val_ratio` consistent with the training run that produced the checkpoint.

Example:

```bash
python -m src.evaluate \
  --split val \
  --model baseline \
  --ckpt weights/best_baseline.pt \
  --save_overlays \
  --save_confusion_png \
  --save_confusion_npy \
  --save_metrics_json
```

### Test Evaluation

Test evaluation reads the nested `dataset/test` directory directly.

Example:

```bash
python -m src.evaluate \
  --split test \
  --model LogicH \
  --ckpt weights/best_LogicH.pt \
  --test_root dataset/test \
  --save_overlays \
  --save_confusion_png \
  --save_confusion_npy \
  --save_metrics_json
```

### Evaluation Metrics

The evaluation script reports:

- `mIoU`
- `Dice`
- `BoxIoU`
- `DetAcc@0.5`
- `ClsAcc`
- `MacroF1`

It also prints the confusion matrix to the console.

### Evaluation Outputs

Outputs are saved under:

```text
results/<split>/<model>/
```

Depending on flags, this directory may contain:

- prediction overlays
- `confusion_matrix.png`
- `confusion_matrix.npy`
- `metrics.json`

### Overlay Output

When `--save_overlays` is enabled:

- the script saves up to 2 overlays per gesture
- it tries to prefer different clips for the two saved examples

Each overlay contains:

- predicted segmentation mask in red
- ground-truth box in green
- predicted box in blue
- ground-truth gesture label
- predicted gesture label

## Visualization Script

`src/visualise.py` builds side-by-side visual summaries from saved overlay results.

It:

- reads saved overlay images from `results/test/...` and `results/val/...`
- randomly selects two gesture-clip examples per split
- generates comparison figures for both `baseline` and `LogicH`
- writes combined figures to `results/visualise/`

Run it with:

```bash
python -m src.visualise
```

This script assumes that evaluation overlays already exist.

## Convenience Script

`run.sh` contains a simple sequential workflow for:

1. training the baseline model
2. training the `LogicH` model
3. evaluating both models on test and validation splits

Example:

```bash
bash run.sh
```

Current commands in `run.sh` use:

- `50` epochs
- `batch_size=32`
- saved checkpoints in `weights/`
- evaluation output flags enabled

## Metrics Definition

### Segmentation

- `mIoU`: mean intersection over union over all samples
- `Dice`: Dice score averaged over all samples

### Detection / Localization

- `BoxIoU`: IoU between predicted and ground-truth hand boxes
- `DetAcc@0.5`: percentage of samples with box IoU >= 0.5

### Classification

- `ClsAcc`: gesture classification accuracy
- `MacroF1`: macro-averaged F1 score computed from the confusion matrix

## Reproducibility

The utility function in `src/utils.py` sets:

- Python random seed
- NumPy random seed
- PyTorch random seed
- CUDA random seed
- deterministic cuDNN mode

To reproduce a run more reliably, keep the following consistent:

- `--seed`
- `--val_ratio`
- `--image_size`
- `--batch_size`
- model type

## Current Code Assumptions and Notes

### 1. RGB-D Only

The current repository supports RGB-D models only:

- `baseline`
- `LogicH`

There is no RGB-only model path in the current implementation.

### 2. Validation Split Depends on Seed

The validation loader is recreated by random split from the training dataset. If seed or validation ratio changes, the validation subset changes as well.

### 3. Best Model Criterion

The training script saves the best checkpoint using validation `mIoU`, not classification accuracy or a combined score.

### 4. Case Handling for `LogicH`

The CLI is intended to use:

```text
--model LogicH
```

Internally, model names are normalized, so the implementation resolves the correct model branch consistently.

## Example Workflow

### Train Both Models

```bash
python -m src.train --dataset_root dataset --model baseline --epochs 50 --batch_size 32
python -m src.train --dataset_root dataset --model LogicH --epochs 50 --batch_size 32
```

### Evaluate Both Models

```bash
python -m src.evaluate --split test --model baseline --ckpt weights/best_baseline.pt --test_root dataset/test --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json
python -m src.evaluate --split test --model LogicH --ckpt weights/best_LogicH.pt --test_root dataset/test --save_overlays --save_confusion_png --save_confusion_npy --save_metrics_json
```

### Generate Visual Summaries

```bash
python -m src.visualise
```

## Suggested Citation / Project Description

If you need a short project summary for a report, poster, or repository front page, you can use:

> LogicH is an RGB-D multi-task learning framework for hand understanding that jointly performs hand segmentation, bounding box localization, and gesture classification. It compares a lightweight RGB-D baseline against an enhanced LogicH architecture with classification-focused gated cross-modal fusion and depth-guided box refinement.
