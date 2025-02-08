import math

import torch
import torchvision

from dataset import ACTIONS


# Borrowed from https://github.com/pytorch/examples/blob/main/word_language_model/model.py
class PositionalEncoding(torch.nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)


    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class CustomizedViViT(torch.nn.Module):
    def __init__(self, args, num_classes, input_len=10):
        super(CustomizedViViT, self).__init__()

        # self.stride = 2
        # seq_length = seq_length // self.stride

        self.combined_pred = args.combined_pred


        if args.model == "b16":
            self.spatial_model = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)
            self.img_size = 224
            embedding_size = 768
            # param_count: 86_567_656, token_count: 196
        
        elif args.model == "b16se":
            self.spatial_model = torchvision.models.vit_b_16(weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.img_size = 384
            embedding_size = 768
            # param_count: 86_859_496, token_count: 576
        
        elif args.model == "h14se":
            self.spatial_model = torchvision.models.vit_h_14(weights = torchvision.models.ViT_H_14_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.img_size = 518
            embedding_size = 1280
            # param_count: 633_470_440, token_count: 1369
        
        elif args.model == "l16":
            self.spatial_model = torchvision.models.vit_l_16(weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_V1)
            self.img_size = 224
            embedding_size = 1024
            # param_count: 304_326_632, token_count: 196
        
        elif args.model == "l16se":
            self.spatial_model = torchvision.models.vit_l_16(weights = torchvision.models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            self.img_size = 512
            embedding_size = 1024
            # param_count: 305_174_504, token_count: 1024


        # Reset the head layer
        self.spatial_model.heads = torch.nn.Identity()  # torch.nn.Linear(768, temp_emb_size)


        # Learnable positional encoding
        # self.temp_positional_encoding = torch.nn.Parameter(torch.empty(1, input_len, embedding_size).normal_(std=0.02))
        
        # Sinusoidal positional encoding
        self.temp_positional_encoding = PositionalEncoding(embedding_size)

        self.temporal_model = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(d_model=embedding_size, nhead=args.nhead, dim_feedforward=args.dim_feedforward, batch_first=True),
            num_layers=args.num_layers
        )
        if self.combined_pred:
            self.head = torch.nn.Linear(embedding_size, num_classes)
        else:
            self.head_left = torch.nn.Linear(embedding_size, num_classes)
            self.head_right = torch.nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # No need to apply stride in here we already applied it in the dataset. 
        # In the original paper, they both use stride 2 and tubelet size of 16x16x2 and 
        # use central frame inflation which means they practically use stride 4.

        # x = x[:, ::self.stride] 
        
        bs, n_frames, c, h, w = x.size()
        x = x.reshape(bs * n_frames, c, h, w)
        
        x = self.spatial_model(x)

        x = x.reshape(bs, n_frames, -1)

        # x = x + self.temp_positional_encoding
        x = self.temp_positional_encoding(x)
        
        x = self.temporal_model(x)

        if self.combined_pred:
            return self.head(x)
        else:
            left_ac = self.head_left(x)
            right_ac = self.head_right(x)
            return torch.cat([left_ac, right_ac], dim=0)


def make_model(args, device): 

    if args.model in ["b16", "b16se", "l16", "l16se", "h14se"]:
        # My ViViT model using the pretrained ViT weights from torch, model 2 in the paper 
        model = CustomizedViViT(args, len(ACTIONS), input_len=10).to(device=device)
        
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model
