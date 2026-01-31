import tensorflow as tf


class TripletSemiHardLoss(tf.keras.losses.Loss):
    """
    Implementation of Batch-Hard Triplet Loss.
    Optimized for speed (O(N^2)) and stability.
    """

    def __init__(self, margin=0.5, name="triplet_loss"):
        super().__init__(name=name)
        self.margin = margin

    def call(self, y_true, y_pred):
        # --- 1. Robust Label Handling ---
        # Fixes the 'int64 vs int32' error you saw
        labels = tf.cast(y_true, dtype=tf.int32)
        if len(labels.shape) == 2:
            labels = tf.squeeze(labels, axis=1)

        # --- 2. Robust Feature Handling ---
        # Ensure embeddings are float32 and normalized
        embeddings = tf.cast(y_pred, tf.float32)
        embeddings = tf.math.l2_normalize(embeddings, axis=1)

        # --- 3. Efficient Distance Matrix (Batch x Batch) ---
        # ||a-b||^2 = ||a||^2 - 2<a,b> + ||b||^2
        # Since ||a||=1, this simplifies to: 2 - 2<a,b>
        dot_product = tf.matmul(embeddings, embeddings, transpose_b=True)
        distances = 2.0 - 2.0 * dot_product
        distances = tf.maximum(distances, 0.0)  # Clip negative float errors

        # --- 4. Masking ---
        # mask_pos: Pairs with SAME label (excluding diagonal)
        # mask_neg: Pairs with DIFFERENT label
        label_equal = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
        mask_pos = tf.cast(label_equal, tf.float32) - tf.eye(
            tf.shape(labels)[0], dtype=tf.float32
        )
        mask_neg = 1.0 - tf.cast(label_equal, tf.float32)

        # --- 5. Batch Hard Mining (Max Pos - Min Neg) ---
        # Hardest Positive: The one furthest away
        hardest_pos = tf.reduce_max(distances * mask_pos, axis=1)

        # Hardest Negative: The one closest (we add infinity to invalid ones to ignore them)
        max_dist_in_batch = tf.reduce_max(distances)
        distances_for_min = distances + max_dist_in_batch * (1.0 - mask_neg)
        hardest_neg = tf.reduce_min(distances_for_min, axis=1)

        # --- 6. Loss Calculation ---
        loss = tf.maximum(hardest_pos - hardest_neg + self.margin, 0.0)
        return tf.reduce_mean(loss)
