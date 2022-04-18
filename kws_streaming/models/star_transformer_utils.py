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

class StarTransformer(tf.keras.Model):
    r"""
    Star-Transformer 的encoder部分。 输入3d的文本输入, 返回相同长度的文本编码
    paper: https://arxiv.org/abs/1902.09113
    """

    def __init__(self, hidden_size, num_layers, num_patches, num_head, head_dim, dropout=0.1, max_len=None):
        r"""

        :param int hidden_size: 输入维度的大小。同时也是输出维度的大小。
        :param int num_layers: star-transformer的层数
        :param int num_head: head的数量。
        :param int head_dim: 每个head的维度大小。
        :param float dropout: dropout 概率. Default: 0.1
        :param int max_len: int or None, 如果为int，输入序列的最大长度，
            模型会为输入序列加上position embedding。
            若为`None`，忽略加上position embedding的步骤. Default: `None`
        """
        super(StarTransformer, self).__init__()
        self.iters = num_layers
        self.hidden_size = hidden_size

        self.norm = [LayerNormalization(epsilon=1e-6) for _ in range(self.iters)]
        self.emb_drop = Dropout(dropout)
        # self.ring_att = [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
        #      for _ in range(self.iters)]
        # self.star_att = [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
        #      for _ in range(self.iters)]
        # self.patch_proj = Dense(hidden_size, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD),
        #                         bias_initializer=Zeros(), input_shape=(98, 40,))

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, hidden_size, num_patches, 1), initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD))
        # self.ffn = tf.keras.Sequential(
        #     [
        #         Dense(hidden_size, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), bias_initializer=Zeros()),
        #         tfa.layers.GELU(approximate=False),
        #         Dense(hidden_size, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD),
        #               bias_initializer=Zeros()),
        #         Dropout(dropout),
        #     ]
        # )
        self.fc = tf.keras.layers.Conv2D(hidden_size, 1)
        self.enc_layers = [
            star_TransformerBlock(hidden_size, num_layers, num_head, head_dim)
            for _ in range(num_layers)
        ]
        #self.fc = Dense(hidden_size, kernel_initializer=TruncatedNormal(mean=0., stddev=TRUNC_STD), use_bias=False)
    def call(self, data, training=1):
        r"""
        :param FloatTensor data: [batch, length, hidden] 输入的序列
        :param ByteTensor mask: [batch, length] 输入序列的padding mask, 在没有内容(padding 部分) 为 0,
            否则为 1
        :return: [batch, length, hidden] 编码后的输出序列
                [batch, hidden] 全局 relay 节点, 详见论文
        """
        #data = self.patch_proj(data)

        def norm_func(f, x):
            # B, H, L, 1
            return tf.transpose(f(tf.transpose(x,[0,2,3,1]), training=training),[0,3,1,2])

        B, L, H = [data.shape[i] for i in range(3)] #_, num_time_windows, num_freqs = net.shape
        #mask = (mask.eq(False))  # flip the mask for masked_fill_
        #smask = tf.concat([tf.zeros([B, 1],tf.uint8).to(mask), mask], 1)
        embs = tf.transpose(data,[0,2,1])[:, :, :, None]

        #embs = embs + self.pos_emb
        embs = norm_func(self.emb_drop, embs)

        embs = tf.transpose(embs,[0,2,3,1])# x: (B,L,1,H)
        embs = self.fc(embs)
        embs = tf.transpose(embs, [0, 3, 1, 2])
        hidden_size = self.hidden_size
        embs = embs + self.pos_emb
        nodes = embs
        relay = tf.reduce_mean(embs,2,keepdims=True)
        #ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        #r_embs = tf.reshape(embs,[B, hidden_size, 1, L]) #进入star-Transformer前token的embedding，原论文中nodes的更新得与这个做attention
        # for i in range(self.iters):
        #     #ax = tf.concat([r_embs, tf.tile(relay, (1, 1, 1, L))], 2)
        #     ax = tf.tile(relay,(1,1,1,L))
        #     nodes = tf.nn.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
        #     relay = tf.nn.leaky_relu(self.star_att[i](relay, tf.concat([relay, nodes], 2)))
        #     #nodes = nodes.masked_fill_(ex_mask, 0)
        for i, layer in enumerate(self.enc_layers):
            nodes = norm_func(self.norm[i], nodes)
            nodes, relay = layer(nodes, relay, training)
        nodes = tf.transpose(tf.reshape(nodes,[B, hidden_size, L]),[0, 2, 1])
        relay = tf.reshape(relay,[B, hidden_size])

        y = 0.5 * (relay + tf.reduce_max(nodes,1))
        #output = self.ffn(y)  # [bsz, n_cls]
        return y

class _MSA1(tf.keras.layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small

        self.WQ = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WK = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WV = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WO = tf.keras.layers.Conv2D(nhid, 1)

        self.drop = Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def call(self, x, ax=None, training=1):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = x.shape

        x = tf.transpose(x,[0,2,3,1])# x: (B,L,1,H)
        q, k, v = self.WQ(x), self.WK(x), self.WV(x)
        q = tf.transpose(q, [0, 3, 1, 2])
        k = tf.transpose(k, [0, 3, 1, 2])
        v = tf.transpose(v, [0, 3, 1, 2])# q,k,v: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[2]

            ax = tf.transpose(ax, [0, 2, 3, 1])
            ak = self.WK(ax)
            av = self.WV(ax)
            ak = tf.transpose(ak, [0, 3, 1, 2])
            av = tf.transpose(av, [0, 3, 1, 2])
            ak = tf.reshape(ak, [B, nhead, head_dim, aL, L])
            av = tf.reshape(av, [B, nhead, head_dim, aL, L])

        q = tf.reshape(q,[B, nhead, head_dim, 1, L])
        k_unfold = tf.image.extract_patches(images=k, sizes=[1, 1, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                                 padding='SAME')
        k1 = tf.transpose(k_unfold, [0, 1, 3, 2])
        k = tf.reshape(k1, [B, nhead, head_dim, unfold_size, L])
        v_unfold = tf.image.extract_patches(images=v, sizes=[1, 1, 3, 1], strides=[1, 1, 1, 1], rates=[1, 1, 1, 1],
                                                 padding='SAME')
        v1 = tf.transpose(v_unfold, [0, 1, 3, 2])
        v = tf.reshape(v1, [B, nhead, head_dim, unfold_size, L])

        if ax is not None:
            k = tf.concat([k, ak], 3)
            v = tf.concat([v, av], 3)

        alphas = self.drop(tf.nn.softmax(tf.reduce_sum(q * k, 2, keepdims=True) / NP.sqrt(head_dim), 3), training=training)  # B N L 1 U
        att = tf.reshape(tf.reduce_sum(alphas * v, 3), [B, nhead * head_dim, L, 1])

        att = tf.transpose(att, [0, 2, 3, 1])
        ret = self.WO(att)
        ret = tf.transpose(ret, [0, 3, 1, 2])

        return ret

class _MSA2(tf.keras.layers.Layer):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA2, self).__init__()
        self.WQ = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WK = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WV = tf.keras.layers.Conv2D(nhead * head_dim, 1)
        self.WO = tf.keras.layers.Conv2D(nhid, 1)
        self.drop = Dropout(dropout)
        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def call(self, x, y, mask=None, training=1):
        # x: B H 1 1  relay
        # y: B H L 1  nodes
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, L, _ = y.shape

        x = tf.transpose(x, [0, 2, 3, 1])
        y = tf.transpose(y, [0, 2, 3, 1])
        q, k, v = self.WQ(x), self.WK(y), self.WV(y)
        q = tf.transpose(q, [0, 3, 1, 2])# q:(B,H,1,1)
        k = tf.transpose(k, [0, 3, 1, 2])
        v = tf.transpose(v, [0, 3, 1, 2])# k,v: (B,H,L,1)

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

class star_TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_size, num_layers, num_head, head_dim, dropout=0.1):
        super(star_TransformerBlock, self).__init__()
        self.ring_att = _MSA3(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
        self.star_att = _MSA4(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
        self.WQ = tf.keras.layers.Conv2D(num_head * head_dim, 1)
        self.WK = tf.keras.layers.Conv2D(num_head * head_dim, 1)
        self.WV = tf.keras.layers.Conv2D(num_head * head_dim, 1)
        self.hidden_size, self.num_head, self.head_dim, self.unfold_size = hidden_size, num_head, head_dim, 3

    def call(self, x, y, training=1):
        B, H, L, _ = x.shape #x是nodes，y是relay
        x_y = tf.concat([x,y],2)
        x_y = tf.transpose(x_y, [0, 2, 3, 1])
        q, k, v = self.WQ(x_y), self.WK(x_y), self.WV(x_y)
        q = tf.transpose(q, [0, 3, 1, 2])# q:(B,H,1,1)
        k = tf.transpose(k, [0, 3, 1, 2])
        v = tf.transpose(v, [0, 3, 1, 2])# k,v: (B,H,L,1)

        q_relay = tf.expand_dims(q[:, :, -1, :],3)
        k_relay = tf.expand_dims(k[:, :, -1, :],3)
        v_relay = tf.expand_dims(v[:, :, -1, :],3)
        q_nodes = q[:, :, 0:-1, :]
        k_nodes = k[:, :, 0:-1, :]
        v_nodes = v[:, :, 0:-1, :]
        nodes = tf.nn.leaky_relu(self.ring_att(q_nodes, k_nodes, v_nodes, ax_k=k_relay, ax_v=v_relay))
        relay = tf.nn.leaky_relu(self.star_att(q_relay, tf.concat([k_nodes, k_relay], 2), tf.concat([v_nodes, v_relay], 2)))
        return nodes, relay


if __name__ == '__main__':
    print('a')
    data = tf.constant(NP.array(range(7840)).reshape(2, 98, 40), dtype='float32')
    B, L, H = [data.shape[i] for i in range(3)]

    def norm_func(f, x):
        # B, H, L, 1
        # return f(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return tf.transpose(f(tf.transpose(x, [0, 2, 3, 1])), [0, 3, 1, 2])

    hidden_size = 192
    num_head = 3
    head_dim = 64
    iters = 2
    dropout = 0.1

    ring_att = [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
                for _ in range(iters)]
    star_att = [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.0)
                for _ in range(iters)]

    switch_hole = 2

    if switch_hole == 0 : #_MASA1 only
        embs = tf.transpose(data, [0, 2, 1])[:, :, :, None]
        nodes = tf.nn.leaky_relu(ring_att[0](embs))

    if switch_hole == 1 : #_MASA1 and _MASA2
        embs = tf.transpose(data, [0, 2, 1])[:, :, :, None]
        relay = tf.reduce_mean(embs,2,keepdims=True)
        #ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = tf.reshape(embs,[B, H, 1, L])
        ax = tf.concat([r_embs, tf.tile(relay,(1,1,1,L))], 2)
        nodes = tf.nn.leaky_relu(ring_att[0](embs))
        relay = tf.nn.leaky_relu(star_att[0](relay, tf.concat([relay, nodes], 2)))

    if switch_hole == 2 : #the hole StarTransformer
        encoder = StarTransformer(hidden_size=hidden_size,
                                       num_layers=iters,
                                       num_head=num_head,
                                       head_dim=head_dim,
                                       dropout=dropout,)
        nodes, relay = encoder(data)

    print('B')
