import tensorflow as tf
import numpy as np
from sklearn.model_selection import KFold
from tensorflow.keras.callbacks import EarlyStopping
from modules.utils import (
    load_cedar_data,
    MetaEpisodeGenerator,
    get_or_download_dataset,
    ValEERCallback,
)
from modules.networks import build_feature_extractor, build_metric_generator

CONFIG = {
    "LOCAL_PATH": "datasets/cedar/signatures",
    "KAGGLE_HANDLE": "shreelakshmigp/cedardataset",
    "PRETRAINED_WEIGHTS": "01_densenet_weights.h5",
    "FOLDS": 5,
    "EPOCHS": 30,
    "LR": 1e-4,
}


def main():
    print("=== STEP 2: METRIC LEARNING EVALUATION (5-FOLD CV) ===")
    path = get_or_download_dataset(CONFIG["LOCAL_PATH"], CONFIG["KAGGLE_HANDLE"])
    X, y, writers, _ = load_cedar_data(path)

    kf = KFold(n_splits=CONFIG["FOLDS"], shuffle=True, random_state=42)
    u_writers = np.unique(writers)
    accs, aucs, eers = [], [], []

    for fold, (tr_idx, te_idx) in enumerate(kf.split(u_writers)):
        print(f"\n>>> FOLD {fold+1}/{CONFIG['FOLDS']}")
        tf.keras.backend.clear_session()

        tr_w, te_w = u_writers[tr_idx], u_writers[te_idx]
        mask_tr, mask_te = np.isin(writers, tr_w), np.isin(writers, te_w)

        # Load Pretrained Backbone & Freeze
        feat_ext = build_feature_extractor()
        feat_ext.load_weights(
            CONFIG["PRETRAINED_WEIGHTS"], by_name=True, skip_mismatch=True
        )
        feat_ext.trainable = False

        # Train Metric Generator
        meta_model = build_metric_generator(feat_ext)
        meta_model.compile(
            optimizer=tf.keras.optimizers.Adam(CONFIG["LR"]),
            loss="binary_crossentropy",
            metrics=["accuracy", "AUC"],
        )

        val_gen = MetaEpisodeGenerator(X[mask_te], y[mask_te], writers[mask_te])

        callbacks = [
            # NEW: Monitor EER on the Validation Generator
            ValEERCallback(val_gen, mode="score"),
            EarlyStopping(
                monitor="val_eer", mode="min", patience=5, restore_best_weights=True
            ),
        ]

        hist = meta_model.fit(
            MetaEpisodeGenerator(X[mask_tr], y[mask_tr], writers[mask_tr]),
            validation_data=val_gen,
            epochs=CONFIG["EPOCHS"],
            callbacks=callbacks,
            verbose=1,
        )

        # We can extract the best EER from the history if we logged it, or just take the final
        best_eer = min(hist.history["val_eer"])
        best_acc = max(hist.history["val_accuracy"])
        best_auc = max(hist.history["val_auc"])

        accs.append(best_acc)
        aucs.append(best_auc)
        eers.append(best_eer)
        print(
            f"   [FOLD {fold+1} RESULT] Acc: {best_acc:.4f} | AUC: {best_auc:.4f} | EER: {best_eer:.4f}"
        )

    print("\n" + "=" * 40)
    print(f"FINAL THESIS RESULTS ({CONFIG['FOLDS']}-Fold CV)")
    print(f"Avg Accuracy: {np.mean(accs):.4f}")
    print(f"Avg ROC-AUC:  {np.mean(aucs):.4f}")
    print(f"Avg EER:      {np.mean(eers):.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
