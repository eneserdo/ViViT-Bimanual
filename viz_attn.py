import pickle
import random
from os.path import join as opj

import matplotlib.pyplot as pyplot
import matplotlib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from model import make_model
from dataset import get_bimanual_dataset_splits
from utils import arg_parser_viz_attn as arg_parser

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pyplot.ioff()
matplotlib.use('Agg')

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)


def get_sample_batch(args):
    TRAIN_SUBSET = [150, 500, 1500]
    VAL_SUBSET = [50, 100, 200]
    
    train_set, test_set, val_set = get_bimanual_dataset_splits(args)


    with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
        sample_dict = pickle.load(f)


    samples = []
    labels = []
    names = []
    for i in TRAIN_SUBSET:
        samples.append(train_set[i][0])
        labels.append(train_set[i][1])
        sb, task, take, fr, act = sample_dict["train"][i]
        names.append(f"{sb}{task}{take}_{fr}")
    

    for i in VAL_SUBSET:
        samples.append(val_set[i][0])
        labels.append(val_set[i][1])
        sb, task, take, fr, act = sample_dict["val"][i]
        names.append(f"{sb}{task}{take}_{fr}")
    
    
    return torch.stack(samples), labels, names


ATTENTION = {"s": [], "t": []}
def getAttentionHook(name):
    def hook(model, input, output):
        # assert isinstance(model, torch.nn.MultiheadAttention)

        num_heads = model.num_heads
        head_dim = model.head_dim

        x = input[0].detach() # y = mha(x, x, x)

        if model.batch_first:
            x = x.transpose(1, 0)

        seq_len, bs, embed_dim = x.shape

        wq, wk, wv = torch.split(model.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0)   
        bq, bk, bv = torch.split(model.in_proj_bias, [embed_dim, embed_dim, embed_dim], dim=0)

        q = torch.matmul(x, wq.T) + bq
        k = torch.matmul(x, wk.T) + bk
        v = torch.matmul(x, wv.T) + bv

        q = q.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)
        k = k.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)
        v = v.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)

        scale = 1.0 / np.sqrt(head_dim)

        attn=torch.bmm(q, k.transpose(1, 2))
        attn=torch.softmax(attn*scale, dim=-1)

        attn_reshaped = attn.reshape(bs, num_heads, seq_len, seq_len)
        # attn_reshaped_avr = attn_reshaped.mean(dim=1)
        # attn_reshaped_max = attn_reshaped_avr.max(dim=1).values

        # double check
        # y = torch.bmm(attn, v)
        # y = y.transpose(0, 1).reshape(seq_len*bs, embed_dim)
        # y = y @ model.out_proj.weight.T + model.out_proj.bias
        # y = y.reshape(seq_len, bs, embed_dim)
        # if model.batch_first:
        #     y = y.transpose(1, 0)
        # # Make sure you set the dropout to 0.0
        # assert torch.allclose(output[0], y, atol=1e-7) # Sometimes it fails due to numerical precision somehow, so set it somethign like 1e-4

        ATTENTION[name[0]].append(attn_reshaped)


        
    return hook


def set_hooks(model):
    # spatial attention
    for i in range(len(model.spatial_model.encoder.layers)):
        model.spatial_model.encoder.layers[i].self_attention.register_forward_hook(getAttentionHook(f"s_layer_{i}"))
    
    # temporal attention
    for i in range(len(model.temporal_model.layers)):
        model.temporal_model.layers[i].self_attn.register_forward_hook(getAttentionHook(f"t_layer_{i}"))


def save_attention_maps(save_dir, attention_maps, names):
    for i, (name, attention_map) in enumerate(attention_maps.items()):
        for j in range(attention_map.shape[0]):
            pyplot.imshow(attention_map[j].cpu().numpy())
            pyplot.savefig(opj(save_dir, f"{names[j]}_{name}_{i}.png"))


def rollout(attentions, discard_ratio, head_fusion, has_cls_token=True):
    # attentions: list of layers, 
    # FIXME: this code assumes bs=1
    bs, num_token = attentions[0].shape[:2]
    result = torch.eye(num_token)

    with torch.no_grad():
        for attention in attentions:
            if head_fusion == "mean":
                attention_heads_fused = attention.mean(axis=1)
            elif head_fusion == "max":
                attention_heads_fused = attention.max(axis=1)[0]
            elif head_fusion == "min":
                attention_heads_fused = attention.min(axis=1)[0]
            else:
                raise "Attention head fusion type Not supported"
            

            # Drop the lowest attentions, but
            # don't drop the class token
            flat = attention_heads_fused.view(bs, -1)
            _, indices = flat.topk(int(flat.size(-1)*discard_ratio), -1, False)

            if has_cls_token:
                for idx, item in enumerate(indices):
                    item = item[item != 0]
                    flat[idx, item] = 0
            else:
                flat[indices] = 0

            Identity = torch.eye(attention_heads_fused.size(-1))
            a = (attention_heads_fused + 1.0*Identity)/2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)
    
    return result


def load_imgs(names):
    ...

def viz_attn(spacial_rollout_attn, temporal_rollout_attn, original_images, pred, labels, names):
    # Look at the total attention between the class token,
    # and the image patches
    # mask = result[0, 0, 1:]
    # # In case of 224x224 image, this brings us from 196 to 14
    # width = int(mask.size(-1)**0.5)
    # mask = mask.reshape(width, width).numpy()
    # mask = mask / np.max(mask)

    ...

def main(args):    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.gpu_id))


    model = make_model(args, device)
    # model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    print("Currently no model is loaded".upper())

    set_hooks(model)

    samples, labels, names = get_sample_batch(args)    

    model.eval()
    with torch.no_grad():
        # load model
        pred = model(samples.to(device))
        pred = pred.cpu().numpy()


    spatial_rollout_attn = rollout(ATTENTION["s"], 0.5, "mean") # her bir vid için 16 tane ?

    temporal_rollout_attn = rollout(ATTENTION["t"], 0.5, "mean", False)
    original_images = load_imgs(names)

    viz_attn(spatial_rollout_attn, temporal_rollout_attn, original_images, pred, labels, names)
    # pick middle frame's label




    
        
            


if __name__ == "__main__":
    args = arg_parser()
    main(args)