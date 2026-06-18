# Identify Doodles

Identify Doodles is a lightweight deep learning project for recognizing hand-drawn doodles. The system uses a MobileNetV1-style convolutional neural network and an interactive Pygame canvas so users can draw a sketch and receive an immediate prediction.

## Supported Classes

The current model is designed for 10 doodle categories:

- airplane
- apple
- banana
- car
- cat
- duck
- fish
- hand
- house
- soccer ball

## Project Highlights

- MobileNetV1-based classifier optimized for fast inference.
- Depthwise separable convolution blocks to reduce computation while preserving useful visual features.
- Training pipeline with image resizing, augmentation, train/validation split, checkpoint saving, and training history.
- Interactive Pygame demo for drawing and classifying doodles.
- Checkpoint metadata stores class labels to keep training and inference label order consistent.

## Repository Structure

```text
Identify-Doodles/
+-- Train.py                 # Training pipeline
+-- src/
|   +-- Main.py              # Pygame inference application
|   +-- model/
|       +-- Model.py         # MobileNetV1 model definition
+-- images/
|   +-- Icon.png             # App/static image asset
+-- README.md
```

Large local files are intentionally ignored by Git:

- `QuickDrawDataset/`
- `QuickDrawDataset.zip`
- `*.pt` model checkpoints
- generated temporary images
- report/submission documents in the local report folder

## Dataset

The dataset follows the `ImageFolder` layout, where each class is stored in a separate folder:

```text
QuickDrawDataset/
+-- airplane/
+-- apple/
+-- banana/
+-- car/
+-- cat/
+-- duck/
+-- fish/
+-- hand/
+-- house/
+-- soccer ball/
```

The local training set used for this project contains 30,000 images, with 3,000 images per class. The dataset is not included in this repository because it is large. Download or prepare the dataset separately, then place it at `QuickDrawDataset/` or update `DATA_DIR` in `Train.py`.

## Model Architecture

The classifier follows the MobileNetV1 design:

```text
Input image
    |
    v
Initial 3x3 convolution
    |
    v
Depthwise separable convolution blocks
    |
    v
Average pooling
    |
    v
Fully connected classifier
    |
    v
Predicted doodle class
```

MobileNetV1 is a good fit for doodle recognition because the task depends mostly on shape, stroke direction, and object structure. The depthwise separable convolution design keeps the model efficient enough for interactive use.

## Training

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python Train.py
```

Training uses:

- input size: `224 x 224`
- batch size: `32`
- epochs: `30`
- validation split: `20%`
- optimizer: `Adam`
- loss function: `CrossEntropyLoss`
- learning-rate scheduler: `ReduceLROnPlateau`

The best checkpoint is saved as `best_checkpoint.pt`. Checkpoint files are ignored by Git, so keep them locally or upload them separately as release assets if needed.

## Running the Demo

Place a trained checkpoint named `best_checkpoint.pt` in the project root, then run:

```bash
python src/Main.py
```

The application opens a drawing canvas. After a sketch is submitted, the image is preprocessed and passed through the trained model. The predicted class is displayed in the interface.

## Current Result

The current best local checkpoint reached approximately `98.72%` validation accuracy on the prepared 10-class QuickDraw subset.

Validation accuracy may be higher than real drawing performance because user-created sketches can differ from the dataset in stroke thickness, centering, scale, and drawing style.

## Recommended Improvements

- Standardize preprocessing for user-drawn images, including bounding-box crop, centering, padding, and stroke normalization.
- Add top-k predictions with confidence scores.
- Add a confusion matrix and per-class precision/recall/F1 report.
- Expand the number of supported doodle classes.
- Package the demo or provide a web interface for easier testing.

## Git Notes

This repository is configured to keep source code and lightweight assets in Git while excluding datasets, checkpoints, generated files, and report documents. If a large file was tracked before `.gitignore` was added, remove it from Git tracking while keeping it locally:

```bash
git rm --cached <file>
```
