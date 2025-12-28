import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import os
import numpy as np
import timm
import cv2

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "dataset", "vggface2_112x112")
DB_FILE = os.path.join(BASE_DIR, "face_db.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definitions
MODEL_CONFIGS = {
    "EfficientNet": {
        "path": os.path.join(BASE_DIR, "models", "lookalike_epoch_effnet_10.pth"),
        "color": "#1f77b4"
    },
    "ResNet50": {
        "path": os.path.join(BASE_DIR, "models", "lookalike_resnet_epoch_10.pth"),
        "color": "#2ca02c"
    },
    "ViT-Base": {
        "path": os.path.join(BASE_DIR, "models", "lookalike_vit_epoch_10.pth"),
        "color": "#d62728"
    }
}


# --- 1. METRICS HELPER ---
def calculate_metrics(user_emb, db_emb):
    import torch.nn.functional as F
    if isinstance(user_emb, np.ndarray): user_emb = torch.from_numpy(user_emb)
    if isinstance(db_emb, np.ndarray): db_emb = torch.from_numpy(db_emb)
    if len(user_emb.shape) == 1: user_emb = user_emb.unsqueeze(0)
    if len(db_emb.shape) == 1: db_emb = db_emb.unsqueeze(0)

    l2_dist = F.pairwise_distance(user_emb, db_emb).item()
    cosine_sim = F.cosine_similarity(user_emb, db_emb).item()
    confidence = max(0, cosine_sim) * 100
    return {"Euclidean Dist": l2_dist, "Cosine Sim": cosine_sim, "Confidence": confidence}


# --- 2. MODEL ARCHITECTURES ---
class SiameseNetwork_EffNet(nn.Module):
    def __init__(self):
        super(SiameseNetwork_EffNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        input_feats = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(input_feats, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class SiameseNetwork_ResNet(nn.Module):
    def __init__(self):
        super(SiameseNetwork_ResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)
        self.backbone.maxpool = nn.Identity()
        input_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(input_features, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        return torch.nn.functional.normalize(x, p=2, dim=1)


class SiameseNetwork_ViT(nn.Module):
    def __init__(self):
        super(SiameseNetwork_ViT, self).__init__()
        self.backbone = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=0)
        self.classifier = nn.Sequential(
            nn.Linear(768, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Linear(512, 128)
        )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.backbone(x)
        embedding = self.classifier(features)
        return torch.nn.functional.normalize(embedding, p=2, dim=1)


# --- 3. HELPERS ---
class FaceHandler:
    def __init__(self):
        from facenet_pytorch import MTCNN
        self.mtcnn = MTCNN(keep_all=False, device=DEVICE, margin=0, min_face_size=40, select_largest=True)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def detect_and_embed(self, image_pil, model):
        try:
            boxes, _ = self.mtcnn.detect(image_pil)
        except:
            return None, None, None
        if boxes is not None:
            box = boxes[0]
            if box[0] < 0 or box[1] < 0 or box[2] > image_pil.width or box[
                3] > image_pil.height: return None, None, None
            box_safe = [max(0, b) for b in box]
            face = image_pil.crop(box_safe)
            face_resized = face.resize((112, 112), Image.BILINEAR)
            t = self.transform(face_resized).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                emb = model(t).cpu().numpy()
            return box, emb, face_resized
        return None, None, None

    def process_static_image(self, image_pil):
        # Auto-skip if image is small (Dataset image)
        w, h = image_pil.size
        if w <= 224 and h <= 224:
            img_resized = image_pil.resize((112, 112), Image.BILINEAR)
            t = self.transform(img_resized)
            return t, img_resized, image_pil

        # Otherwise detect
        try:
            boxes, _ = self.mtcnn.detect(image_pil)
        except:
            return None, None, image_pil
        if boxes is not None:
            box = boxes[0]
            draw = ImageDraw.Draw(image_pil)
            draw.rectangle(box.tolist(), outline="#00FF00", width=6)
            box_safe = [max(0, b) for b in box]
            face = image_pil.crop(box_safe)
            face = face.resize((112, 112), Image.BILINEAR)
            return self.transform(face), face, image_pil
        return None, None, image_pil


# --- 4. CACHED LOADERS ---
@st.cache_resource
def load_resources():
    models = {}
    for name, cfg in MODEL_CONFIGS.items():
        try:
            if name == "EfficientNet":
                m = SiameseNetwork_EffNet().to(DEVICE)
            elif name == "ResNet50":
                m = SiameseNetwork_ResNet().to(DEVICE)
            elif name == "ViT-Base":
                m = SiameseNetwork_ViT().to(DEVICE)

            if os.path.exists(cfg["path"]):
                m.load_state_dict(torch.load(cfg["path"], map_location=DEVICE))
                m.eval()
                models[name] = m
            else:
                st.sidebar.error(f"Missing {name}: {cfg['path']}")
        except Exception as e:
            st.sidebar.error(f"Error {name}: {e}")

    # --- DB LOADING FIX ---
    # We added weights_only=False to allow loading dictionaries/lists safely
    if os.path.exists(DB_FILE):
        print(f"⚡ Loading pre-computed database from {DB_FILE}...")
        try:
            saved_data = torch.load(DB_FILE, weights_only=False)
            return models, saved_data["embeddings"], saved_data["paths"], saved_data["ids"]
        except Exception as e:
            st.error(f"Failed to load DB file: {e}. Please delete face_db.pt and restart.")
            return models, None, None, None

    if not os.path.exists(DATA_PATH): return models, None, None, None

    print("🐢 No saved DB found. Indexing dataset...")
    transform = transforms.Compose([
        transforms.Resize((112, 112)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    emb_map = {name: [] for name in models.keys()}
    paths = []
    ids = []

    identities = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    subset = identities  # Full dataset

    progress_bar = st.progress(0)
    status = st.empty()

    for i, identity in enumerate(subset):
        folder = os.path.join(DATA_PATH, identity)
        files = os.listdir(folder)
        if files:
            img_path = os.path.join(folder, files[0])
            try:
                img = Image.open(img_path).convert('RGB')
                t = transform(img).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    for name, model in models.items():
                        emb_map[name].append(model(t).cpu().numpy())
                paths.append(img_path)
                ids.append(identity)
            except:
                pass

        if i % 100 == 0:
            progress_bar.progress(i / len(subset))
            status.text(f"Indexing {i}/{len(subset)}...")

    progress_bar.empty()
    status.empty()

    final_emb = {k: np.vstack(v) for k, v in emb_map.items() if v}

    print(f"💾 Saving database to {DB_FILE}...")
    torch.save({"embeddings": final_emb, "paths": paths, "ids": ids}, DB_FILE)

    return models, final_emb, paths, ids


# --- 5. MAIN UI ---
def main():
    st.set_page_config(page_title="LookAlike Finder", layout="wide")
    st.title("👁️ LookAlike: Multimodal Face Recognition")

    with st.spinner("Loading Engines..."):
        models, db_embs, db_paths, db_ids = load_resources()

    if not models or not db_embs:
        st.error("Setup failed. Check logs.")
        return

    face_handler = FaceHandler()

    def render_dashboard(img_pil):
        t_face, face_crop, boxed_img = face_handler.process_static_image(img_pil)
        if t_face is None: st.error("No face detected."); return

        with st.container():
            col_input, col_output = st.columns([1, 3])
            with col_input:
                st.subheader("Input")
                boxed_img.thumbnail((400, 400))
                st.image(boxed_img, caption="Detected Face", width=300)
                st.image(face_crop, caption="AI Input (112px)", width=112)

            with col_output:
                st.subheader("Model Predictions")
                m_cols = st.columns(3)
                t_face = t_face.unsqueeze(0).to(DEVICE)

                for idx, (name, model) in enumerate(models.items()):
                    if name not in db_embs: continue
                    with torch.no_grad():
                        user_emb = model(t_face).cpu().numpy()

                    sims = np.dot(db_embs[name], user_emb.T).flatten()
                    best_idx = np.argmax(sims)

                    matched_emb = db_embs[name][best_idx]
                    metrics = calculate_metrics(user_emb, matched_emb)

                    match_path = db_paths[best_idx]
                    match_id = db_ids[best_idx].replace('n', '')

                    with m_cols[idx]:
                        color = MODEL_CONFIGS.get(name, {}).get("color", "black")
                        st.markdown(f"<h4 style='color: {color}; border-bottom: 2px solid {color};'>{name}</h4>",
                                    unsafe_allow_html=True)
                        st.image(Image.open(match_path), caption=f"ID #{match_id}", width="stretch")
                        st.metric("Confidence", f"{metrics['Confidence']:.1f}%")
                        st.text(f"L2 Dist: {metrics['Euclidean Dist']:.3f}")
                        st.text(f"Cos Sim: {metrics['Cosine Sim']:.3f}")

    tab1, tab2, tab3 = st.tabs(["📁 Upload", "📸 Snapshot", "🎥 FaceScan"])

    with tab1:
        f = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        if f: render_dashboard(Image.open(f).convert('RGB'))

    with tab2:
        c = st.camera_input("Take Photo")
        if c: render_dashboard(Image.open(c).convert('RGB'))

    with tab3:
        st.subheader("Real-time Face Recognition")
        st.caption("Running with EfficientNet")
        threshold = st.slider("Strictness Threshold", 0.0, 1.0, 0.70, 0.01)

        # Use Checkbox to maintain state better than Button in some cases
        run_scan = st.checkbox("Turn on Webcam")

        if run_scan:
            scan_model = models.get("EfficientNet")
            # FIX: CAP_DSHOW fixes the MSMF error on Windows
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

            if not cap.isOpened():
                st.error("Could not open webcam. Ensure Zoom/Teams is closed.")
            else:
                st_frame = st.empty()
                while run_scan:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read frame.")
                        break

                    # Convert BGR (OpenCV) to RGB (PIL)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb)

                    box, emb, _ = face_handler.detect_and_embed(pil_img, scan_model)

                    if box is not None:
                        sims = np.dot(db_embs["EfficientNet"], emb.T).flatten()
                        best_idx = np.argmax(sims)
                        best_sim = sims[best_idx]

                        x1, y1, x2, y2 = [int(c) for c in box]

                        if best_sim > threshold:
                            clean_id = db_ids[best_idx].replace("n", "")
                            label = f"ID #{clean_id} ({best_sim:.2f})"
                            col = (0, 255, 0)
                        else:
                            label = "Unknown"
                            col = (255, 0, 0)

                        cv2.rectangle(rgb, (x1, y1), (x2, y2), col, 3)
                        cv2.putText(rgb, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)

                    st_frame.image(rgb)
                cap.release()

    st.markdown("---")
    st.markdown("Made with ❤️ by Abdelbasset Moujtahid")


if __name__ == "__main__":
    main()