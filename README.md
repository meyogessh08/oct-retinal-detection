# Retinal Disease Detection Using Deep Learning on OCT Images

Classifies retinal OCT scans into **CNV, DME, DRUSEN, NORMAL** with PyTorch, compares a custom CNN
against ImageNet-pretrained ResNet18 / VGG16 / MobileNetV2 / DenseNet121, explains predictions
with Grad-CAM, and serves everything through a Streamlit app.

> **Disclaimer:** educational prototype, not a clinical diagnostic tool.

## Project structure

```
oct-retinal-detection/
├── app.py                  # Streamlit app
├── requirements.txt
├── src/
│   ├── config.py           # all settings (paths, sizes, class names)
│   ├── preprocessing.py    # loading, balanced subsets, splits, transforms, DataLoaders
│   ├── models.py           # custom CNN + pretrained models + checkpoint helpers
│   ├── train.py            # training loop (L2, early stopping, LR scheduler)
│   ├── evaluate.py         # metrics, confusion matrices, model comparison table
│   ├── gradcam.py          # Grad-CAM from scratch
│   ├── perceptron_numpy.py # NumPy neural-network fundamentals
│   └── explore.py          # EDA: class counts, sample images
├── notebooks/kaggle_runner.py   # copy cells into a Kaggle notebook
└── outputs/                # models/, figures/, results/
```

## Setup (laptop, for coding + running the app)

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On a CPU-only laptop, install the CPU build of PyTorch first if pip's default is slow:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

## Training (Kaggle GPU)

Training happens on Kaggle, see `notebooks/kaggle_runner.py` for the exact cells. Short version:

```bash
python -m src.explore                                              # EDA
python -m src.perceptron_numpy                                     # NumPy fundamentals
python -m src.train --model custom_cnn --epochs 15 --lr 1e-3
python -m src.train --model resnet18 --mode feature_extraction --epochs 6 --lr 1e-3
python -m src.train --model resnet18 --mode fine_tune --epochs 8 --lr 1e-4
python -m src.train --model mobilenet_v2 --mode fine_tune
python -m src.train --model densenet121 --mode fine_tune
python -m src.train --model vgg16 --mode feature_extraction --epochs 4 --lr 1e-3
python -m src.evaluate                                             # comparison table + confusion matrices
```

## Run the app

Copy the chosen checkpoint to `outputs/models/best_model.pt`, then:

```bash
streamlit run app.py
```

## Methodology notes

* **Class imbalance:** the raw training split is imbalanced, so a class-balanced subset is sampled
  (`TRAIN_PER_CLASS` in `config.py`).
* **Leakage:** validation is carved from the train pool with disjoint indices. The Hugging Face
  version has no patient IDs, so patient-level splitting is impossible and validation accuracy may be
  optimistic; the official test split is the unbiased number.
* **Augmentation:** mild only (small shifts, brightness/contrast, horizontal flip). No vertical flips
  because OCT anatomy has a fixed orientation.
* **Limitation:** softmax confidence is not a reliability measure; out-of-distribution images can
  still receive a confident prediction. The app adds basic input checks but cannot fully prevent this.

## Results

_Fill in after training: paste `outputs/results/model_comparison.csv` here._
