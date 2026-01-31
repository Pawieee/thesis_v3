import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    GlobalAveragePooling2D,
    GlobalMaxPooling2D,
    Reshape,
    Multiply,
    Add,
    Lambda,
    Concatenate,
    BatchNormalization,
    Dropout,
    Conv2D,
    Input,
)
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import Model


def cbam_block(x, ratio=8):
    c = int(x.shape[-1])
    av = Dense(c, activation="sigmoid")(
        Dense(c // ratio, activation="relu")(
            Reshape((1, 1, c))(GlobalAveragePooling2D()(x))
        )
    )
    mx = Dense(c, activation="sigmoid")(
        Dense(c // ratio, activation="relu")(
            Reshape((1, 1, c))(GlobalMaxPooling2D()(x))
        )
    )
    x = Multiply()([x, Add()([av, mx])])
    concat = Concatenate(axis=-1)(
        [
            Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True))(x),
            Lambda(lambda t: tf.reduce_max(t, axis=-1, keepdims=True))(x),
        ]
    )
    return Multiply()([x, Conv2D(1, 7, padding="same", activation="sigmoid")(concat)])


def build_feature_extractor(input_shape=(224, 224, 3), embedding_dim=512):
    base = DenseNet121(weights="imagenet", include_top=False, input_shape=input_shape)
    for layer in base.layers[:-30]:
        layer.trainable = False
    x = cbam_block(base.output)
    x = Dropout(0.3)(BatchNormalization()(GlobalAveragePooling2D()(x)))
    embeddings = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(
        Dense(embedding_dim)(x)
    )
    return Model(inputs=base.input, outputs=embeddings, name="FeatureExtractor")


def build_metric_generator(feature_extractor):
    ia, ib = Input((224, 224, 3)), Input((224, 224, 3))
    combined = Concatenate()([feature_extractor(ia), feature_extractor(ib)])
    x = Dropout(0.4)(BatchNormalization()(Dense(256, activation="relu")(combined)))
    x = BatchNormalization()(Dense(128, activation="relu")(x))
    return Model(
        inputs=[ia, ib],
        outputs=Dense(1, activation="sigmoid")(x),
        name="MetricGenerator",
    )
