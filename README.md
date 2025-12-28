# LookAlike

LookAlike is a small, self-contained face recognition and "lookalike" search project built with PyTorch and Streamlit. It uses Siamese-style models (EfficientNet, ResNet50 and ViT) trained to produce 128-d normalized face embeddings and offers an interactive web UI to upload an image, take a snapshot, or run real-time webcam scanning.

## Key features
- Three model backbones supported: EfficientNet (EffNet-B0), ResNet50 and ViT-Base. Pretrained .pth files are stored in `models/`.
- Uses MTCNN (via `facenet_pytorch`) for face detection and automatic cropping.
- Embeddings are normalized and compared with a pre-computed database (`face_db.pt`) created from the VGGFace2 112x112 dataset.
- Streamlit-based web UI with three tabs: Upload, Snapshot (camera input), and FaceScan (webcam real-time inference).
- Lightweight metric helpers for Euclidean distance, cosine similarity and a confidence heuristic.

## Table of contents
- Features
- Quickstart
- Requirements
- Installation
- Running the app
- Downloading the dataset
- File structure
- How it works
- Troubleshooting
- Notes & next steps

## Quickstart
1. (Recommended) Create and activate a virtual environment.
2. Install dependencies (see the Requirements section below).
3. If you have not yet downloaded the dataset, run `python download.py` to fetch VGGFace2 (the script uses `kagglehub`).
4. Start the Streamlit app:

    ```cmd
    streamlit run app.py
    ```

## Requirements
- Python 3.8+
- PyTorch (install the version appropriate for your platform and CUDA from https://pytorch.org)
- torchvision
- timm
- facenet-pytorch
- streamlit
- opencv-python
- pillow
- numpy
- kagglehub (optional; used by `download.py` to fetch dataset)

Notes on PyTorch: The project expects a matching PyTorch + CUDA configuration if you want GPU acceleration. Install PyTorch separately using the recommendations on the official website for correct CUDA support.

## Installation (Windows - cmd.exe)
1. Create and activate a virtual env:

    ```cmd
    python -m venv .venv
    .venv\Scripts\activate
    ```

2. Install Python packages (this project provides a minimal `requirements.txt` — install PyTorch separately if needed):

    ```cmd
    python -m pip install -r requirements.txt
    ```

3. Install PyTorch (example CPU-only install; replace with the CUDA command from https://pytorch.org if you have an NVIDIA GPU):

    ```cmd
    python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    ```

## Running the app
- Start the Streamlit UI:

    ```cmd
    streamlit run app.py
    ```

- Open the URL printed in the terminal (usually `http://localhost:8501`).
- Use the "Upload" tab to upload an image file, "Snapshot" to take a photo with your camera, and "FaceScan" to enable webcam-based real-time recognition.

## Downloading the dataset
- `download.py` wraps `kagglehub` to download the VGGFace2 (112x112) dataset and moves it to `./dataset`. You must configure Kaggle credentials or `kagglehub` as required by that tool. Example:

    ```cmd
    python download.py
    ```

- Alternatively, place a dataset with the same structure under `dataset/vggface2_112x112/`. The app will index the dataset and generate `face_db.pt` automatically (this can take some time depending on dataset size).
- If `face_db.pt` is present in the repo root it will be loaded directly to speed startup.

## Model weights
- The repository contains example model checkpoints in the `models/` folder:
  - `lookalike_epoch_effnet_10.pth`
  - `lookalike_resnet_epoch_10.pth`
  - `lookalike_vit_epoch_10.pth`

If any of these files are missing the app will show a sidebar error and skip that engine.

## File structure overview
- `app.py` — Streamlit app, model wrappers and main UI logic.
- `download.py` — Helper to download VGGFace2 with `kagglehub`.
- `metrics.py` — Utility functions for computing distances, cosine similarity and confidence heuristics.
- `models/` — Pretrained model checkpoint files (.pth).
- `dataset/` — Expected dataset root (VGGFace2 112x112 layout: `id_xxx/` subfolders with images).
- `face_db.pt` — Serialized embeddings, paths and ids (created by the app when indexing the dataset).

## How it works (brief)
1. The app loads available model backbones and (if present) a saved `face_db.pt` containing precomputed embeddings.
2. When an image is provided, MTCNN detects the face, crops and resizes to 112x112.
3. The model produces a 128-d normalized embedding for the detected face.
4. The embedding is compared to the database embeddings (dot product / cosine). The best match is shown along with basic metrics and a heuristic confidence value.

## Troubleshooting
- Webcam won't open or shows errors: on Windows the app uses the CAP_DSHOW flag to avoid MSMF issues; ensure other apps (Zoom, Teams) are not holding the camera.
- Missing models: copy your `.pth` files into `models/` or retrain/export compatible weights.
- face_db.pt load errors: delete `face_db.pt` to force re-indexing the dataset. The app will re-create it automatically.
- MTCNN issues: `facenet_pytorch` depends on a working PyTorch + torchvision setup; verify CUDA/CPU compatibility and that the package is installed.


## Author & Source
Made with ❤️ by Abdelbasset Moujtahid
If you find this project useful, consider sourcing the [paper](./ICML_PMML_Report_amoujt2s.pdf)
```
@inproceedings{
    title={LookAlike - Open-Set Face Retrieval Using Deep Metric Learning},
    author={Moujtahid, Abdelbasset},
    url={https://github.com/moujtahid21/LookAlike},
    booktitle={Project Report for EECS6327 (Fall 2025), Department of Electrical Engineering and Computer Science, York University},
    year={2025}   
}
```
or starring the GitHub repository.

## License
Project Report for EECS6327 (Fall 2025), Department of Electrical Engineering and Computer Science, YorkUniversity. 
Copyright 2025 by the author(s).