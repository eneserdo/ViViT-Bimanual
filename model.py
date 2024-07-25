import torch
import torchvision
import transformers

from dataset import ACTIONS

class FramewiseHead(torch.nn.Module):
    def __init__(self, num_classes):
        super(FramewiseHead, self).__init__()
        self.left_fc = torch.nn.Linear(768, num_classes)
        self.right_fc = torch.nn.Linear(768, num_classes)

    
    def forward(self, x):
        bs = x.size(0)
        x = x[:, 1:, :] # remove the first token, which is the CLS token
        x = x.reshape(bs, 16, 14, 14, 768) # FIXME: hardcoded shape
        x = x.mean(dim=[2, 3])
        left = self.left_fc(x)
        right = self.right_fc(x)

        return torch.cat([left, right], dim=0)

# Just to unify the model usage
class HFViViT(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.net = transformers.VivitModel.from_pretrained("google/vivit-b-16x2-kinetics400")
        self.net.pooler = FramewiseHead(len(ACTIONS))
    
    def forward(self, x):
        return self.net(x).pooler_output


class MyViViT(torch.nn.Module):
    def __init__(self, args, num_classes, seq_length=32):
        super(MyViViT, self).__init__()

        self.stride = 2
        seq_length = seq_length // self.stride
        # Note that this already includes the positional encoding and class token
        # if args.img_size == 384:
        #     self.spatial_model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        # elif args.img_size == 224:

        self.spatial_model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1)

        # Reset the head layer
        self.spatial_model.heads = torch.nn.Identity()  # torch.nn.Linear(768, temp_emb_size)


        self.temp_positional_encoding = torch.nn.Parameter(torch.empty(1, seq_length, 768).normal_(std=0.02))

        self.temporal_model = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(d_model=768, nhead=args.nhead, dim_feedforward=args.dim_feedforward, batch_first=True),
            num_layers=args.num_layers
        )

        self.head_left = torch.nn.Linear(768, num_classes)
        self.head_right = torch.nn.Linear(768, num_classes)

    def forward(self, x):
        # To simplify the code, I apply stride=2 in here instead of creating tubelets of size 16x16x2
        x = x[:, ::self.stride] 
        
        bs, n_frames, c, h, w = x.size()
        x = x.reshape(bs * n_frames, c, h, w)
        
        x = self.spatial_model(x)

        x = x.reshape(bs, n_frames, -1)

        x = x + self.temp_positional_encoding
        
        x = self.temporal_model(x)

        left_ac = self.head_left(x)
        right_ac = self.head_right(x)

        return torch.cat([left_ac, right_ac], dim=0)


def make_model(args):
    assert torch.cuda.is_available()
    device = torch.device("cuda") 

    if args.model == "hf":
        # From huggingface, model 1 in the paper
        model = HFViViT(len(ACTIONS)).to(device=device)


    elif args.model == "tr":
        # My ViViT model using the pretrained ViT weights from torch, model 2 in the paper 
        model = MyViViT(args, len(ACTIONS), seq_length=32).to(device=device)
        
    else:
        raise ValueError(f"Unknown model: {args.model}")

    return model
