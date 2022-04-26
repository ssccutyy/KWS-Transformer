# coding=utf-8
# Copyright (c) 2021, Arm Limited and Contributors.
# SPDX-License-Identifier: Apache-2.0
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Vision transformer implementation based on https://github.com/tuvovan/Vision_Transformer_Keras/blob/master/vit.py

import numpy as NP
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization,
    LayerNormalization,
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling

from tensorflow.keras.initializers import Zeros, Ones, TruncatedNormal, Constant
from tensorflow.python.ops import math_ops

TRUNC_STD = 0.02

class _MSA3(tf.keras.layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA3, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small

        self.WO = tf.keras.layers.Conv2D(nhid, 1)

        self.drop = Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def call(self, q, k, v, ax_k=None, ax_v=None, training=1):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = k.shape
        aL = ax_k.shape[2]
        ax_k = tf.tile(ax_k, (1, 1, 1, L))
        ax_v = tf.tile(ax_v, (1, 1, 1, L))
        ak = tf.reshape(ax_k, [B, nhead, head_dim, aL, L])
        av = tf.reshape(ax_v, [B, nhead, head_dim, aL, L])

        q = tf.reshape(q,[B, nhead, head_dim, 1, L])
        k_unfold = tf.image.extract_patches(images=k, sizes=[1, 1, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                                 padding='SAME')
        k1 = tf.transpose(k_unfold, [0, 1, 3, 2])
        k = tf.reshape(k1, [B, nhead, head_dim, unfold_size, L])
        v_unfold = tf.image.extract_patches(images=v, sizes=[1, 1, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                                 padding='SAME')
        v1 = tf.transpose(v_unfold, [0, 1, 3, 2])
        v = tf.reshape(v1, [B, nhead, head_dim, unfold_size, L])

        if ax_k is not None:
            k = tf.concat([k, ak], 3)
            v = tf.concat([v, av], 3)

        alphas = self.drop(tf.nn.softmax(tf.reduce_sum(q * k, 2, keepdims=True) / NP.sqrt(head_dim), 3), training=training)  # B N L 1 U
        att = tf.reshape(tf.reduce_sum(alphas * v, 3), [B, nhead * head_dim, L, 1])

        att = tf.transpose(att, [0, 2, 3, 1])
        ret = self.WO(att)
        ret = tf.transpose(ret, [0, 3, 1, 2])

        return ret

class _MSA4(tf.keras.layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA4, self).__init__()
        self.WO = tf.keras.layers.Conv2D(nhid, 1)
        self.drop = Dropout(dropout)
        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def call(self, q, k, v, mask=None, training=1):
        # x: B H 1 1  relay
        # y: B H L 1  nodes
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = k.shape

        q = tf.reshape(q,[B, nhead, 1, head_dim])  # B, H, 1, 1 -> B, N, 1, h
        k = tf.reshape(k,[B, nhead, head_dim, L])  # B, H, L, 1 -> B, N, h, L
        v = tf.transpose(tf.reshape(v,[B, nhead, head_dim, L]),[0, 1, 3, 2])# B, H, L, 1 -> B, N, L, h
        pre_a = tf.matmul(q, k) / NP.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(tf.nn.softmax(pre_a, 3), training=training)  # B, N, 1, L
        att = tf.reshape(tf.matmul(alphas, v),[B, -1, 1, 1])  # B, N, 1, h -> B, N*h, 1, 1

        att = tf.transpose(att, [0, 2, 3, 1])
        ret = self.WO(att)
        ret = tf.transpose(ret, [0, 3, 1, 2])
        return ret

class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        self.key_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        self.value_dense = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
        #self.combine_heads = Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros())
        self.ring_att = _MSA3(embed_dim, nhead=num_heads, head_dim=self.projection_dim, dropout=0.0)
        self.star_att = _MSA4(embed_dim, nhead=num_heads, head_dim=self.projection_dim, dropout=0.0)

    def call(self, x, y):
        B, H, L, _ = x.shape  # x是nodes，y是relay
        x_y = tf.concat([x, y], 2)
        x_y = tf.transpose(x_y, [0, 2, 3, 1])
        q, k, v = self.query_dense(x_y), self.key_dense(x_y), self.value_dense(x_y)
        q = tf.transpose(q, [0, 3, 1, 2])  # q:(B,H,1,1)
        k = tf.transpose(k, [0, 3, 1, 2])
        v = tf.transpose(v, [0, 3, 1, 2])  # k,v: (B,H,L,1)

        q_relay = tf.expand_dims(q[:, :, -1, :], 3)
        k_relay = tf.expand_dims(k[:, :, -1, :], 3)
        v_relay = tf.expand_dims(v[:, :, -1, :], 3)
        q_nodes = q[:, :, 0:-1, :]
        k_nodes = k[:, :, 0:-1, :]
        v_nodes = v[:, :, 0:-1, :]
        nodes = self.ring_att(q_nodes, k_nodes, v_nodes, ax_k=k_relay, ax_v=v_relay)
        relay = self.star_att(q_relay, tf.concat([k_nodes, k_relay], 2), tf.concat([v_nodes, v_relay], 2))
        return nodes, relay

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, prenorm=False, approximate_gelu=False):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)

        self.ffn = tf.keras.Sequential(
            [
                Dense(ff_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
                tfa.layers.GELU(approximate=approximate_gelu),
                Dense(embed_dim, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
            ]
        )

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.prenorm = prenorm


    def call(self, nodes_i, relay_i, training):
        if self.prenorm:
            nodes = self.layernorm1(nodes_i)
            nodes, relay = self.att(nodes, relay_i)
            nodes = self.dropout1(nodes, training=training)
            nodes = nodes + nodes_i
            relay = relay + relay_i
            nodes = self.layernorm2(nodes)
            y = tf.concat([relay,nodes], axis=2)
            y = tf.transpose(y,[0,3,2,1])
            y = self.ffn(y)
            y = tf.transpose(y, [0, 3, 2, 1])
            nodes_o = y[:, :, 1:, :]
            relay_o = tf.expand_dims(y[:, :, 0, :],3)
            nodes_o = nodes + self.dropout2(nodes_o, training=training)
            relay_o = relay_o + relay
        else:
            nodes, relay = self.att(nodes_i, relay_i)
            nodes = self.dropout1(nodes, training=training)
            nodes = nodes + nodes_i
            relay = relay + relay_i
            nodes = self.layernorm1(nodes)
            y = tf.concat([relay, nodes], axis=2)
            y = tf.transpose(y, [0, 3, 2, 1])
            y = self.ffn(y)
            y = tf.transpose(y, [0, 3, 2, 1])
            nodes_o = y[:, :, 1:, :]
            relay_o = tf.expand_dims(y[:, :, 0, :],3)
            nodes_o = self.dropout2(nodes_o, training=training)
            nodes_o = nodes_o + nodes
            relay_o = relay_o + relay
            nodes_o = self.layernorm2(nodes_o)
        return nodes_o, relay_o

class Star2Transformer(tf.keras.Model):
    def __init__(
        self,
        num_patches,
        num_layers,
        num_classes,
        d_model,
        num_heads,
        mlp_dim,
        channels=3,
        dropout=0.1,
        prenorm=False,
        distill_token=False,
        approximate_gelu=False,
    ):
        super(Star2Transformer, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD))
        #self.class_emb = self.add_weight("class_emb", shape=(1, 1, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD))
        #self.distill_emb = self.add_weight("distill_emb", shape=(1, 1, d_model), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD)) if distill_token else None
        self.patch_proj = Dense(d_model, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros(), input_shape=(98,40,))

        self.enc_layers = [
            TransformerBlock(d_model, num_heads, mlp_dim, dropout, prenorm, approximate_gelu)
            for _ in range(num_layers)
        ]


    def extract_patches(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches


    def call(self, x, training):
        B, L, H = x.shape
        x = self.patch_proj(x)

        x = x + self.pos_emb

        nodes = x
        relay = tf.reduce_mean(x,1,keepdims=True)
        nodes = tf.transpose(nodes, [0, 2, 1])[:, :, :, None]
        relay = tf.transpose(relay, [0, 2, 1])[:, :, :, None]
        for layer in self.enc_layers:
            nodes, relay = layer(nodes, relay, training)

        nodes = tf.transpose(tf.reshape(nodes, [B, self.d_model, L]), [0, 2, 1])
        relay = tf.reshape(relay, [B, self.d_model])
        y = 0.5 * (relay + tf.reduce_max(nodes,1))

        return y


if __name__ == '__main__':
    print("a")
    net = tf.constant(NP.array(range(7840)).reshape(2, 98, 40), dtype='float32')

    time_transformer = KWSTransformer(num_layers=1,
        num_classes=12,
        d_model=192,
        num_heads=3,
        mlp_dim=768,
        dropout=0.0,
        num_patches=98,
        prenorm=True,
        distill_token=False,
        approximate_gelu=False,
        )

    time_sig = time_transformer(net, training=1)
    print("b")
