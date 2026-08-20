# Wood Board Knots Instance Segmentation

This repository contains the complete pipeline for training, evaluating, and running inference for 
instance segmentation of wood board knots using YOLOv11.

The project is structured to support a modular MLOps workflow, cleanly separating data processing, 
model training, evaluation metrics, and future C++ ONNX/TensorRT deployment.

## Project Structure

- `configs/` – YAML configuration files for datasets and training parameters.
- `data/` – (Ignored by Git) Directory for raw and processed datasets.
- `src/` – Python source code for data splitting, training, and evaluation.
- `scripts/` – CLI entry points for pipeline execution.
- `runs/` – (Ignored by Git) Ultralytics output artifacts (weights, logs, plots).

## Installation

1. Clone the repository.
2. Create a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Dataset Preparation

Place your raw images and YOLO-format segmentation labels into `data/raw/`.

Run the split script to organize the data into train/val/test sets while preserving board frame 
continuity:

```bash
python src/data/split_dataset.py
```

Ensure the generated `data/processed/` structure matches the paths defined in `configs/data.yaml`.

## Training the Baseline Model

To train the YOLOv11 segmentation baseline model, run:

```bash
python src/training/train_baseline.py
```

Model checkpoints and training metrics will be saved automatically to `runs/segment/baseline/`.

## Inference

To run inference and visualize the predicted polygons on new images:

```bash
# Example CLI command (to be implemented in scripts/infer.py)
yolo predict model=runs/segment/baseline/weights/best.pt source=data/processed/test/images
```
