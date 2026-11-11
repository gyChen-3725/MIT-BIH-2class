"""将作者 notebook 的 BAFNet 网络按相同层参数整理为可导入的 Keras 模型。"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2


class ScaledDotProductAttention(layers.Layer):
    """对应原 notebook 的无掩码缩放点积注意力。"""

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        query, key, value = inputs
        scale = tf.sqrt(tf.cast(tf.shape(query)[-1], query.dtype))
        weights = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / scale, axis=-1)
        return tf.matmul(weights, value)


def _conv(filters: int, stride: int = 1) -> layers.Layer:
    return layers.Conv1D(
        filters, 11, strides=stride, padding="same", activation="relu",
        kernel_initializer="he_normal", kernel_regularizer=l2(1e-3),
        bias_regularizer=l2(1e-3),
    )


def _decoder(features: tf.Tensor, name: str) -> tf.Tensor:
    features = layers.UpSampling1D(5)(features)
    features = layers.Conv1DTranspose(
        24, 11, padding="same", kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
    )(features)
    features = layers.UpSampling1D(3)(features)
    features = layers.Conv1DTranspose(
        16, 11, strides=2, padding="same", kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3),
    )(features)
    return layers.Conv1DTranspose(
        1, 11, padding="same", kernel_initializer="he_normal",
        kernel_regularizer=l2(1e-3), bias_regularizer=l2(1e-3), name=name,
    )(features)


def create_model() -> keras.Model:
    """900×2 输入、三路输出及层宽均与作者的 ``create_model`` 对应。"""

    inputs = layers.Input(shape=(900, 2))
    rri = layers.Reshape((900, 1))(inputs[:, :, 0])
    amp = layers.Reshape((900, 1))(inputs[:, :, 1])
    rri = _conv(16)(rri)
    amp = _conv(16)(amp)
    rri = layers.MaxPooling1D(3, padding="same")(_conv(24, 2)(rri))
    amp = layers.MaxPooling1D(3, padding="same")(_conv(24, 2)(amp))
    fused = layers.Concatenate(axis=-1, name="fsn2")([rri, amp])

    rri = layers.MaxPooling1D(5, padding="same")(_conv(32)(rri))
    amp = layers.MaxPooling1D(5, padding="same")(_conv(32)(amp))
    fused = layers.MaxPooling1D(5, padding="same")(_conv(32)(fused))
    fused = ScaledDotProductAttention()([fused, fused, fused])
    rri = ScaledDotProductAttention()([fused, rri, rri])
    amp = ScaledDotProductAttention()([fused, amp, amp])
    joined = layers.Concatenate(axis=-1, name="Concat_Layer_x1")([rri, amp])

    rri_reconstruction = _decoder(rri, "rri")
    amp_reconstruction = _decoder(amp, "ampl")
    scale = layers.GlobalAveragePooling1D()(joined)
    scale = layers.Dense(32, activation="relu")(scale)
    scale = layers.Dense(64, activation="sigmoid")(scale)
    scale = layers.Reshape((1, 64))(scale)
    fused_output = layers.Multiply()([joined, scale])
    fused_output = layers.GlobalAveragePooling1D(name="GAP")(fused_output)
    predictions = layers.Dense(2, activation="softmax", name="outputs")(fused_output)
    return keras.Model(inputs, [predictions, rri_reconstruction, amp_reconstruction])
