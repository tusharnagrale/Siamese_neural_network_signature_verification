# 🖊️ Signature Detection & Verification

This repository provides an end-to-end pipeline for **signature detection** using YOLO and **signature verification** using a Siamese Neural Network (SNN).

---

## 📌 Setup & Annotation with Label Studio

Install **Label Studio**:

```bash
pip install label-studio
```

Start Label Studio with local file sharing enabled:

```bash
LOCAL_FILES_SERVING_ENABLED=true label-studio start
```

After annotating your dataset, visualize YOLO annotations:

```bash
python3 visualize_yolo_labeling.py
```

This will generate annotated images for all images in the directory.

---

## 📌 YOLO Model Training (Signature Detector)

### 1️⃣ Create a virtual environment
```bash
python -m venv .venv
```

Activate it:

```bash
.venv/Scripts/activate   # Windows
source .venv/bin/activate  # Mac/Linux
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Split dataset into train/validation sets

Expected input structure:
```
data/
├── images/
│   ├── 0b155f33-7a78-4ad2-8f13-fecf2c6ed84a.jpg
│   ├── 0cad3a72-cb9f-4e6c-884e-1356ed323506.jpg
├── labels/
│   ├── 0b155f33-7a78-4ad2-8f13-fecf2c6ed84a.txt
│   ├── 0cad3a72-cb9f-4e6c-884e-1356ed323506.txt
```

Run the split script:

```bash
python train_val_split.py --datapath="data/" --train_pct=0.9
```

Output structure:
```
data/
├── train/
│   ├── images/
│   └── labels/
└── validation/
    ├── images/
    └── labels/
```

---

### 3️⃣ Train YOLO on Google Colab (with GPU)

1. Upload `yolo_signature_detector.ipynb` to **Google Drive**  
2. Create `data.zip` (containing dataset) and upload to Google Drive  
3. Run notebook cells step by step  

Notebook flow:
- **Cell 3** → Copy `data.zip` from Drive to Colab  
- **Cell 4** → Unzip `data.zip`  
- **Cell 5** → Split into train/validation sets  
- **Cell 6** → Create `data.yaml` for YOLO training  
- **Cell 7** → Train YOLO model (`epochs` configurable)  
  - Model output path:
    ```
    data/runs/detect/train/weights/best.pt
    ```
- **Cell 8** → Create `my_model.zip` and download it locally  

Extract `my_model.zip` into the root directory:

```
SIAMESE_NEURAL_NETWORK_SIGNATURE_VERIFICATION/
```

---

## 📌 Siamese Neural Network (Signature Verification)

### 1️⃣ Download dataset
```bash
python download_cedar_dataset.py
```

Extract the dataset into the root directory.

### 2️⃣ Train the SNN model
```bash
python snn_training.py
```

The trained model will be saved under:

```
artifacts/
├── siamese_best.keras
├── optimal_threshold.json
```
give the path to the models in (signature_verifier.py) file
run the streamlit application 
```bash
streamlit run signature_verifier.py
```

---

## ✅ Summary
- **Label Studio** → Annotation & visualization  
- **YOLO** → Signature detection  
- **SNN** → Signature verification  

This workflow enables an **end-to-end automated signature detection + verification system**.
