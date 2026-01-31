import os
import cv2
import re
import math
import numpy as np
import tensorflow as tf
import kagglehub
from scipy.ndimage import rotate, gaussian_filter
from glob import glob
from sklearn.metrics import roc_curve


# --- 1. METRICS & CALLBACKS (NEW) ---
def calculate_eer(y_true, y_scores):
    """
    Calculates Equal Error Rate (EER) given true labels and raw scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # Find index where fnr and fpr are closest
    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer = fpr[eer_idx]
    return eer


class ValEERCallback(tf.keras.callbacks.Callback):
    """
    Custom Callback to calculate EER at the end of every epoch.
    Supports both Stage 1 (Distance-based) and Stage 2 (Score-based).
    """

    def __init__(self, validation_data, mode="distance", batch_size=32):
        super().__init__()
        self.validation_data = validation_data
        self.mode = mode  # 'distance' for Stage 1, 'score' for Stage 2
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        if self.mode == "distance":
            # STAGE 1 LOGIC: Predict Embeddings -> Calc Euclidean -> Calc EER
            X_val, y_val = self.validation_data
            embeddings = self.model.predict(
                X_val, batch_size=self.batch_size, verbose=0
            )

            # Exhaustive pair generation (fast for small val sets)
            distances, labels = [], []
            u_writers = np.unique(y_val)

            for w in u_writers:
                # 0-54 are IDs, >=55 are forgeries (if utilizing that label logic)
                # Assuming y_val contains writer IDs.
                # Note: In Stage 1 loader, we just passed raw writer IDs for simple triplet mining.
                # We need to distinguish gen vs forg based on the loader logic.

                # Check if we have forgery labels or just IDs
                # In Stage 1, y usually contains just Writer IDs (0-54).
                # To calculate EER properly in Stage 1, we need to know which images are forgeries.
                # The current Stage 1 loader labels forgeries as (ID + 55).

                g_idxs = np.where((y_val == w))[0]
                f_idxs = np.where((y_val == w + 55))[0]  # Forgery class

                # Genuine Pairs
                for i in range(len(g_idxs)):
                    for j in range(i + 1, len(g_idxs)):
                        d = np.linalg.norm(
                            embeddings[g_idxs[i]] - embeddings[g_idxs[j]]
                        )
                        distances.append(d)
                        labels.append(1)  # Same

                # Forgery Pairs (if forgeries exist in val set)
                if len(f_idxs) > 0:
                    for i in range(len(g_idxs)):
                        for j in range(len(f_idxs)):
                            d = np.linalg.norm(
                                embeddings[g_idxs[i]] - embeddings[f_idxs[j]]
                            )
                            distances.append(d)
                            labels.append(0)  # Different

            if len(labels) == 0:
                return  # Safety

            # For ROC, higher score = positive class.
            # Distance is inverse (lower = better). So we pass -distances.
            eer = calculate_eer(labels, -np.array(distances))

        elif self.mode == "score":
            # STAGE 2 LOGIC: Predict Scores directly
            # validation_data is a Generator here
            y_true, y_pred = [], []
            val_gen = self.validation_data

            # Iterate through the whole validation generator once
            for i in range(len(val_gen)):
                X_batch, y_batch = val_gen[i]
                preds = self.model.predict(X_batch, verbose=0)
                y_true.extend(y_batch)
                y_pred.extend(preds.flatten())

            eer = calculate_eer(y_true, y_pred)

        print(f" - val_eer: {eer:.4f}")
        logs["val_eer"] = eer


# --- 2. DATASET MANAGEMENT ---
def get_or_download_dataset(local_path, kaggle_handle):
    if os.path.exists(local_path) and any(os.scandir(local_path)):
        print(f"[DATA] Found local dataset at: {local_path}")
        return local_path
    print(f"[DATA] Local path missing. Downloading {kaggle_handle}...")
    return kagglehub.dataset_download(kaggle_handle)


# --- 3. PREPROCESSING ---
def preprocess_image_padded(img_path, target_size=(224, 224)):
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = cv2.bitwise_not(thresh)
    h, w = img.shape
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(img, (new_w, new_h))
    top, bottom = (target_size[0] - new_h) // 2, (target_size[0] - new_h) - (
        target_size[0] - new_h
    ) // 2
    left, right = (target_size[1] - new_w) // 2, (target_size[1] - new_w) - (
        target_size[1] - new_w
    ) // 2
    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0
    )
    return cv2.cvtColor(padded, cv2.COLOR_GRAY2RGB).astype("float32") / 255.0


# --- 4. LOADER ---
def load_cedar_data(root_path):
    print(f"[LOADER] Scanning CEDAR at: {root_path}")
    images, labels, writer_ids = [], [], []
    all_files = []
    for ext in ["*.png", "*.jpg", "*.tif"]:
        all_files.extend(glob(os.path.join(root_path, "**", ext), recursive=True))

    id_pattern = re.compile(r"_(\d+)_")
    for f in all_files:
        match = id_pattern.search(os.path.basename(f))
        if match:
            wid = int(match.group(1)) - 1
            is_forg = ("forg" in f.lower()) or ("full_forg" in f.lower())
            img = preprocess_image_padded(f)
            if img is not None:
                images.append(img)
                labels.append(wid + 55 if is_forg else wid)
                writer_ids.append(wid)

    if not images:
        raise ValueError("No images found!")
    return np.array(images), np.array(labels), np.array(writer_ids), 55


# --- 5. GENERATORS ---
class TripletBatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size, k_instances, num_writers, augment=False):
        self.X, self.y = X, y
        self.batch_size, self.k, self.num_writers = batch_size, k_instances, num_writers
        self.augment = augment
        self.indices = {c: np.where(y == c)[0] for c in np.unique(y)}
        self.genuine = [c for c in np.unique(y) if c < num_writers]

    def __len__(self):
        return math.ceil(len(self.X) / self.batch_size)

    def _aug(self, img):
        if np.random.random() < 0.5:
            img = rotate(img, np.random.uniform(-10, 10), reshape=False, mode="nearest")
        if np.random.random() < 0.2:
            img = gaussian_filter(img, sigma=np.random.uniform(0.1, 1.0))
        return np.clip(img, 0, 1)

    def __getitem__(self, idx):
        bx, by = [], []
        # Safely select writers that actually exist in the subset
        available_writers = [w for w in self.genuine if w in self.indices]
        if not available_writers:
            return np.zeros((1, 224, 224, 3)), np.zeros((1,))

        writers = np.random.choice(
            available_writers, max(1, self.batch_size // (2 * self.k)), replace=True
        )
        for w in writers:
            g_idxs = self.indices[w]
            if len(g_idxs) == 0:
                continue
            bx.extend(
                self.X[np.random.choice(g_idxs, self.k, replace=(len(g_idxs) < self.k))]
            )
            by.extend(
                self.y[np.random.choice(g_idxs, self.k, replace=(len(g_idxs) < self.k))]
            )

            f_idxs = self.indices.get(w + self.num_writers, np.array([]))
            for _ in range(self.k):
                if f_idxs.size > 0 and np.random.random() < 0.5:
                    bx.extend(self.X[np.random.choice(f_idxs, 1)])
                    by.extend(self.y[np.random.choice(f_idxs, 1)])
                else:
                    rw = np.random.choice(available_writers)
                    while rw == w:
                        rw = np.random.choice(available_writers)
                    bx.extend(self.X[np.random.choice(self.indices[rw], 1)])
                    by.extend(self.y[np.random.choice(self.indices[rw], 1)])
        bx = np.array(bx)
        if self.augment:
            bx = np.array([self._aug(x) for x in bx])
        return bx, np.array(by)


class MetaEpisodeGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, writer_ids, batch_size=32):
        self.X, self.y, self.ids, self.bs = X, y, writer_ids, batch_size
        self.uids = np.unique(writer_ids)
        self.map = {u: {"g": [], "f": []} for u in self.uids}
        for i, (l, w) in enumerate(zip(y, writer_ids)):
            if l >= 55:
                self.map[w]["f"].append(i)
            else:
                self.map[w]["g"].append(i)

    def __len__(self):
        return 100

    def __getitem__(self, idx):
        pa, pb, lbs = [], [], []
        while len(pa) < self.bs:
            w = np.random.choice(self.uids)
            gens = self.map[w]["g"]
            if len(gens) < 2:
                continue
            s_idx = np.random.choice(gens)

            if np.random.random() < 0.5:
                q_idx = np.random.choice(gens)
                while q_idx == s_idx:
                    q_idx = np.random.choice(gens)
                lbl = 1.0
            else:
                if self.map[w]["f"] and np.random.random() < 0.7:
                    q_idx = np.random.choice(self.map[w]["f"])
                else:
                    ow = np.random.choice(self.uids)
                    while ow == w:
                        ow = np.random.choice(self.uids)
                    q_idx = np.random.choice(self.map[ow]["g"])
                lbl = 0.0
            pa.append(self.X[s_idx])
            pb.append(self.X[q_idx])
            lbs.append(lbl)
        return [np.array(pa), np.array(pb)], np.array(lbs)
