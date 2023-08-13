import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Optional, Tuple
from lowrank import LowRankDense, LowRankTransformerEncoder
from lowformer_utils import ConvBlock, ResidualBlock


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return x


class LowRankViT(torch.nn.Module):
    def __init__(
        self,
        *,
        input_size,
        input_dim=320,
        output_dim=512,
        dim=1024,
        depth=4,
        heads=16,
        mlp_dim=2048,
        pool="cls",
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()

        self.project = LowRankDense(input_dim, dim)

        self.pos_encoder = PositionalEncoding(dim)
        self.pos_embedding = torch.nn.parameter.Parameter(
            torch.randn(1, input_size + 1, dim)
        )

        self.cls_token = torch.nn.parameter.Parameter(torch.randn(1, 1, dim))
        self.dropout = torch.nn.Dropout(emb_dropout)

        self.transformer = LowRankTransformerEncoder(
            dim, depth, heads, mlp_dim, dropout
        )

        self.pool = pool
        self.to_latent = torch.nn.Identity()

        self.mlp_head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), LowRankDense(dim, output_dim)
        )

        self.tanh = torch.nn.Tanh()

    def forward(self, x, mask=None):
        x = self.project(x)

        x = self.pos_encoder(x)

        x = self.dropout(x)

        x = self.transformer(x, mask)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]

        x = self.to_latent(x)

        return self.tanh(self.mlp_head(x))


class Encoder(torch.nn.Module):
    def __init__(
        self,
        num_res_blocks,
        trans_input_channels,
        partials_n_frames,
        model_embedding_size,
    ):
        super().__init__()
        self.num_res_blocks = num_res_blocks
        self.trans_input_channels = trans_input_channels
        self.partials_n_frames = partials_n_frames
        self.model_embedding_size = model_embedding_size

        self.conv_block = ConvBlock(in_channels=34, out_channels=64, kernel_size=3)
        self.residual_blocks = torch.nn.Sequential(
            *[ResidualBlock(channels=64) for i in range(self.num_res_blocks)]
        )

        self.conv_block2 = ConvBlock(in_channels=64, out_channels=64, kernel_size=3)

        self.final_feature = ConvBlock(
            in_channels=64, out_channels=self.trans_input_channels, kernel_size=3
        )

        self.global_avgpool = torch.nn.AvgPool2d(kernel_size=8)

        self.cnn = torch.nn.Sequential(
            *[
                self.conv_block,
                self.residual_blocks,
                self.conv_block2,
                self.final_feature,
                self.global_avgpool,
                torch.nn.Flatten(),
            ]
        )

        # Network defition
        self.transformer = LowRankViT(
            input_size=self.partials_n_frames,
            input_dim=self.trans_input_channels,
            output_dim=self.model_embedding_size,
            dim=1024,
            depth=12,
            heads=8,
            mlp_dim=2048,
            pool="mean",
            dropout=0.0,
            emb_dropout=0.0,
        )

        # Cosine similarity scaling (with fixed initial parameter values)
        self.similarity_weight = torch.nn.parameter.Parameter(torch.tensor([10.0]))
        self.similarity_bias = torch.nn.parameter.Parameter(torch.tensor([-5.0]))

    def forward(self, games, hidden_init=None):
        """
        Computes the embeddings of a batch of games.

        :param games: batch of games of same duration as a tensor of shape
        (batch_size, n_frames, 34, 8, 8)
        :param hidden_init: initial hidden state of the LSTM as a tensor of shape (num_layers,
        batch_size, hidden_size). Will default to a tensor of zeros if None.
        :return: the embeddings as a tensor of shape (batch_size, embedding_size)
        """


        batch_size, n_frames, feature_shape = (
            games.shape[0],
            games.shape[1],
            games.shape[2:],
        )
        
        #  (batch_size, n_frames, 34, 8, 8) -> (batch_size*n_frames, 34, 8, 8)
        games = torch.reshape(games, (batch_size * n_frames, *feature_shape))

        # (batch_size*n_frames, cnn_out_features)
        game_features = self.cnn(games)

        # (batch_size*n_frames, cnn_out_features) -> (batch_size, n_frames, cnn_out_features)
        game_features = torch.reshape(
            game_features, (batch_size, n_frames, game_features.shape[-1])
        )

        # Pass the input into transformer
        # (batch_size, n_frames, n_features)
        embeds_raw = self.transformer(game_features)
        # self.lstm.flatten_parameters()

        # L2-normalize it
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


if __name__ == "__main__":
    model = Encoder(6, 64, 32, 512)
    
    img = torch.zeros((16, 32, 34, 8, 8))
    
    out = model(img)
    print(out.shape)
    print(get_n_params(model))