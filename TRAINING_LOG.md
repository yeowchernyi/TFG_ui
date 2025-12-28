# Project Setup and Training Guide

This document records the steps taken to set up the environment, preprocess data, and train the EGSTalker model.

## 1. Environment Setup

Due to dependency conflicts between `pytorch3d` and `open3d`, we used two separate Conda environments.

### Environment A: `pytorch3d_safe` (For Training)
Used for the main training loop (`train.py`) which requires PyTorch3D and GPU acceleration.

**Installation Steps:**
1.  **Create Environment:**
    ```bash
    conda create -n pytorch3d_safe python=3.8
    conda activate pytorch3d_safe
    ```
2.  **Install PyTorch & CUDA:**
    ```bash
    conda install pytorch==2.4.1 torchvision==0.19.1 pytorch-cuda=11.8 -c pytorch -c nvidia
    ```
3.  **Install PyTorch3D (From Source):**
    *   *Issue:* Standard installation failed due to missing `sqlite3.dll` and compilation errors.
    *   *Fix:* Restored `sqlite3.dll` and installed from local source.
    ```bash
    pip install fvcore iopath
    cd pytorch3d_source
    pip install .
    cd ..
    ```
4.  **Install Dependencies:**
    ```bash
    pip install scikit-learn pandas tensorboard chardet
    pip install matplotlib lpips wandb mmengine mmcv plyfile
    ```
5.  **Install Submodules:**
    ```bash
    pip install submodules/simple-knn
    pip install submodules/custom-bg-depth-diff-gaussian-rasterization
    ```

### Environment B: `open3d_env` (For Preprocessing)
Used for data preprocessing steps that explicitly require Open3D.

**Installation Steps:**
1.  **Create Environment:**
    ```bash
    conda create -n open3d_env python=3.9
    conda activate open3d_env
    ```
2.  **Install Open3D:**
    ```bash
    pip install open3d
    ```

---

## 2. Code Modifications & Fixes

Several modifications were made to the original codebase to resolve compatibility issues.

### A. NumPy 2.0 Compatibility (`data_utils/face_tracking/face_tracker.py`)
*   **Issue:** The `3DMM_info.npy` file was saved with a newer NumPy version, causing `ModuleNotFoundError: No module named 'numpy._core'` when loading in Python 3.8.
*   **Fix:** Added a compatibility patch to alias `numpy._core` to `numpy.core`.

### B. Optional Open3D Import (`utils/point_utils.py`)
*   **Issue:** `train.py` imports `point_utils.py`, which imported `open3d`. This caused crashes in the `pytorch3d_safe` environment where Open3D is not installed.
*   **Fix:** Wrapped the `open3d` import in a `try-except` block to make it optional.

### C. MMCV Config Compatibility (`train.py`)
*   **Issue:** Newer versions of `mmcv` moved the `Config` class to `mmengine`.
*   **Fix:** Updated `train.py` to try importing `Config` from `mmcv` first, and fall back to `mmengine` if that fails.

### D. Missing Data Handling (`data/obama/au.csv`)
*   **Issue:** The `au.csv` file (Action Units for eye blinking) was missing or empty.
*   **Fix:** Created a dummy `au.csv` filled with zeros to allow the training pipeline to proceed without crashing.

---

## 3. Execution Workflow

### Step 1: Data Preprocessing
(Performed in `open3d_env` or `pytorch3d_safe` depending on the specific script requirements)
```bash
python data_utils/process.py data/obama/obama.mp4
```

### Step 2: Training
(Performed in `pytorch3d_safe`)
```bash
conda activate pytorch3d_safe
python train.py -s data/obama --model_path output/obama --configs arguments/args.py
```
