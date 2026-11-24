python -m venv .venv# Kaggle runner: copy each "# %%" block into its own notebook cell.
# Notebook settings (right sidebar): Accelerator = GPU T4 x2 (or P100), Internet = On.

# %% Cell 1 - get your code onto Kaggle
# Option A (recommended): push the repo to GitHub, then clone it:
# !git clone https://github.com/<your-username>/oct-retinal-detection.git
# %cd oct-retinal-detection
# Option B: upload the zip as a Kaggle Dataset and copy it to /kaggle/working.

# %% Cell 2 - install anything missing (torch is preinstalled on Kaggle)
# !pip install -q datasets opencv-python-headless

# %% Cell 3 - confirm GPU
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

# %% Cell 4 - EDA (prints class counts; paste the output back to your mentor)
# !python -m src.explore

# %% Cell 5 - smoke test: tiny run to prove the pipeline works (~1 min)
# !python -m src.train --model resnet18 --mode fine_tune --epochs 1 --train_per_class 200 --tag _smoke

# %% Cell 6 - real training runs (one cell each so a crash doesn't lose everything)
# !python -m src.train --model custom_cnn --epochs 15 --lr 1e-3
# !python -m src.train --model resnet18 --mode feature_extraction --epochs 6 --lr 1e-3
# !python -m src.train --model resnet18 --mode fine_tune --epochs 8 --lr 1e-4

# %% Cell 7 - evaluate everything on the test set
# !python -m src.evaluate

# %% Cell 8 - zip results so you can download them (Output tab -> download)
# !zip -r results.zip outputs/models outputs/figures outputs/results
