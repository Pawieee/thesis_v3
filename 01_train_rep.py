import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from modules.utils import (
    load_cedar_data,
    TripletBatchGenerator,
    get_or_download_dataset,
    ValEERCallback,
)
from modules.networks import build_feature_extractor
from modules.losses import TripletSemiHardLoss

CONFIG = {
    "LOCAL_PATH": "datasets/cedar/signatures",
    "KAGGLE_HANDLE": "shreelakshmigp/cedardataset",
    "BATCH_SIZE": 32,
    "EPOCHS": 50,
    "EMBEDDING_DIM": 512,
    "LR": 1e-4,
}


def main():
    print("=== STEP 1: TRAINING REPRESENTATION (DENSENET + TRIPLET LOSS) ===")
    path = get_or_download_dataset(CONFIG["LOCAL_PATH"], CONFIG["KAGGLE_HANDLE"])
    X, y, writers, NUM_W = load_cedar_data(path)

    # Split
    u_writers = np.unique(writers)
    np.random.shuffle(u_writers)
    tr_w, val_w = u_writers[:45], u_writers[45:]
    print(f"Split: {len(tr_w)} Train Writers | {len(val_w)} Val Writers")

    # Create masks
    mask_tr = np.isin(writers, tr_w)
    mask_val = np.isin(writers, val_w)

    # Generators
    tr_gen = TripletBatchGenerator(
        X[mask_tr], y[mask_tr], CONFIG["BATCH_SIZE"], 4, NUM_W, augment=True
    )
    val_gen = TripletBatchGenerator(
        X[mask_val], y[mask_val], CONFIG["BATCH_SIZE"], 4, NUM_W, augment=False
    )

    model = build_feature_extractor(embedding_dim=CONFIG["EMBEDDING_DIM"])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG["LR"]), loss=TripletSemiHardLoss(0.5)
    )

    callbacks = [
        # NEW: Monitor EER on the Validation Set
        ValEERCallback((X[mask_val], y[mask_val]), mode="distance"),
        # Save model based on EER if possible, otherwise val_loss
        EarlyStopping(
            monitor="val_eer", mode="min", patience=10, restore_best_weights=True
        ),
        ModelCheckpoint(
            "01_densenet_weights.h5", save_best_only=True, monitor="val_eer", mode="min"
        ),
    ]

    model.fit(
        tr_gen, validation_data=val_gen, epochs=CONFIG["EPOCHS"], callbacks=callbacks
    )
    print("[DONE] Step 1 Complete. Weights saved to '01_densenet_weights.h5'")


if __name__ == "__main__":
    main()
