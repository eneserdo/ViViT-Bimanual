# This code is like a personal attack to all programming principles, but it works 

import copy
import os
import pickle
import random
from os.path import join as opj

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import torch
import cv2

from model import make_model
from dataset import get_bimanual_dataset_splits, ACTIONS
from utils import arg_parser_viz_attn as arg_parser

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

plt.ioff()
matplotlib.use('Agg')

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)

SEQ_LEN = 16
# TRAIN_SUBSET = [150, 500, 1500]
# VAL_SUBSET = [50, 100, 200]
TRAIN_SUBSET = [150]
TEST_SUBSET = [50]

def get_sample_batch(args):
    
    train_set, test_set, val_set = get_bimanual_dataset_splits(args)

    with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
        sample_dict = pickle.load(f)

    samples, labels, names, original = [], [], [], []

    for i in TRAIN_SUBSET:
        samples.append(train_set[i][0]) # bu zaten ds oluyor modelin içinde
        original.append(train_set.get_original_imgs(i)[::2]) # Downsample
        labels.append(train_set[i][1][::2])

        sb, task, take, fr, act = sample_dict["train"][i]
        names.append(f"{sb}{task}{take}_{fr}")
    

    for i in TEST_SUBSET:
        samples.append(test_set[i][0]) 
        labels.append(test_set[i][1][::2])
        original.append(test_set.get_original_imgs(i)[::2])

        sb, task, take, fr, act = sample_dict["test"][i]
        names.append(f"{sb}{task}{take}_{fr}")
    
    samples = torch.stack(samples)


    return samples, labels, names, original


ATTENTION = {"s": [], "t": []}
def getAttentionHook(name):
    # print(f"Hook for {name} is set")
    def hook(model, input, output):
        assert isinstance(model, torch.nn.MultiheadAttention)

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

        s_or_t = name[0]
        layer = int(name.split("_")[-1])
        ATTENTION[name[0]][layer].append(attn_reshaped.detach().cpu())

        
    return hook


def set_hooks(model):
    # spatial attention
    for i in range(len(model.spatial_model.encoder.layers)):
        model.spatial_model.encoder.layers[i].self_attention.register_forward_hook(getAttentionHook(f"s_layer_{i}"))
        ATTENTION["s"].append([])
    
    # temporal attention
    for i in range(len(model.temporal_model.layers)):
        model.temporal_model.layers[i].self_attn.register_forward_hook(getAttentionHook(f"t_layer_{i}"))
        ATTENTION["t"].append([])


def rollout(attentions, discard_ratio, head_fusion, has_cls_token=True):
    # attentions: [A_l1, A_l2, ..., A_ln]
    # A_li: [bs, num_heads, num_token, num_token] 
    bs, _, num_token = attentions[0].shape[:3]
    device = attentions[0].device
    result = torch.eye(num_token).to(device)

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

            row_indices = torch.arange(bs).unsqueeze(1).expand(bs, indices.shape[1])

            if has_cls_token: # to preserve the class token (but this is actually cls-cls attention ???)
                first_col = copy.deepcopy(flat[:, 0])
                flat[row_indices, indices] = 0
                flat[:, 0] = first_col
            else:
                flat[row_indices, indices] = 0


            Identity = torch.eye(attention_heads_fused.size(-1)).to(device)
            a = (attention_heads_fused + 1.0*Identity)/2
            a = a / a.sum(dim=-1, keepdim=True)

            result = torch.matmul(a, result)
    
    return result


def get_attentioned_image(images, attns):
    # Look at the total attention between the class token,
    # and the image patches
    im_size = images[0].shape[0]
    attn_size = np.sqrt(attns.shape[-1]).astype(np.int32)
    repeat_count = im_size // attn_size

    ret = []
    for img, attn in zip(images, attns):
        
        # Patches are visible
        mask = attn[0, 1:].reshape(attn_size, attn_size).numpy()
        mask = np.clip(mask / np.max(mask) * 255, 0, 255).astype(np.uint8)
        mask = mask.repeat(repeat_count, axis=0).repeat(repeat_count, axis=1) # 224x224
        mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)[:, :, ::-1]
        combined = cv2.addWeighted(img.numpy(), 0.4, mask, 0.6, 0)
        
        # Alternative: Patches are kinda blurred
        # mask = mask / np.max(mask)
        # mask = cv2.resize(mask, (224, 224))
        # img = np.float32(img) / 255
        # heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)[:, :, ::-1]
        # heatmap = np.float32(heatmap) / 255
        # combined = heatmap + np.float32(img)
        # combined = combined / np.max(combined)
        # combined = np.uint8(255 * combined)

        ret.append(combined)

    return ret


def viz_attn(save_dir, spacial_rollout_attn, temporal_rollout_attn, original_images, preds, labels, names):
    # There are 2N samples (N train, N test), save each of them separately
    # For each sample, there are 16 frames
    # There is two hand for each hand
    selected_frame_idx = 8
    bs_sl, num_tokens, _ = spacial_rollout_attn.shape # [bs*num_frames, num_token, num_token] 
    bs = bs_sl // SEQ_LEN
    spacial_rollout_attn = spacial_rollout_attn.reshape(bs, SEQ_LEN, num_tokens, num_tokens)

    pred_left, pred_right = torch.split(preds, 2, dim=0)
    

    for i in range(len(names)): # for each sample in the batch

        s_attn = spacial_rollout_attn[i] # [SEQ_LEN, num_token, num_token]
        t_attn = temporal_rollout_attn[i] # [SEQ_LEN, SEQ_LEN]
        t_attn = t_attn[selected_frame_idx] # [SEQ_LEN]
        t_attn = (t_attn*100).to(torch.long).tolist()

        original = original_images[i] # [SEQ_LEN, im_size, im_size, 3]
        
        pl_str = [ACTIONS[p][:2] for p in pred_left[i]]
        pr_str = [ACTIONS[p][:2] for p in pred_right[i]]

        gtl_str = [ACTIONS[p][:2] for p in labels[i][:, 0]]
        gtr_str = [ACTIONS[p][:2] for p in labels[i][:, 1]]

        # pl_str = pred_left[i]
        # pr_str = pred_right[i]

        # gtl_str = labels[i][:, 0]
        # gtr_str = labels[i][:, 1]

        subtitles = [f"A{t_attn[j]} t({gtl_str[j]}, {gtr_str[j]}), p({pl_str[j]}, {pr_str[j]})" for j in range(16)]

        n = names[i]

        s_attn_imgs = get_attentioned_image(original, s_attn)

        fig_size = np.ceil(np.sqrt(SEQ_LEN)).astype(np.int32)

        fig, axes = plt.subplots(fig_size, fig_size, figsize=(fig_size*3, fig_size*3))
        axes = axes.flatten()

        for idx, (ax, img, subtitle) in enumerate(zip(axes, s_attn_imgs, subtitles)):
            if idx == selected_frame_idx:
                # put a frame around the selected frame
                background = np.ones((img.shape[0]+10, img.shape[1]+10, 3), dtype=np.uint8) * np.array([255, 0, 0])
                background[5:-5, 5:-5] = img
                img = background  

            ax.imshow(img)
            ax.set_title(subtitle)
            ax.axis("off")
        
        # plt.title(f"Selected frame: {selected_frame_idx}th frame")
        plt.tight_layout()        
        plt.savefig(opj(save_dir, f"attn_{n}.png"))
        plt.close()


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

    save_dir = opj("attention_maps", args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    model = make_model(args, device)
    if args.model_path is not None:
        model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    else:
        print("Currently no model is loaded".upper())

    set_hooks(model)

    samples, labels, names, originals = get_sample_batch(args)
    

    model.eval()
    # with torch.no_grad(): # This let fastpath to work, and it seems fastpath disables the hooks
    # it may not fit to the memory
    preds = []
    for sample in samples:
        preds.append(model(sample.unsqueeze(0).to(device)).detach().cpu().argmax(axis=-1))
    
    preds = torch.cat(preds, dim=0)

    for i in range(len(ATTENTION["s"])):
        ATTENTION["s"][i] = torch.cat(ATTENTION["s"][i], dim=0)
    for i in range(len(ATTENTION["t"])):
        ATTENTION["t"][i] = torch.cat(ATTENTION["t"][i], dim=0)

    spatial_rollout_attn = rollout(ATTENTION["s"], args.discard_ratio, args.fuse_method) # her bir vid için 16 tane ?
    # spatial_rollout_attn: [bs*num_frames, num_token, num_token]

    temporal_rollout_attn = rollout(ATTENTION["t"], args.discard_ratio, args.fuse_method, False)
    # temporal_rollout_attn: [bs, num_token, num_token]
    viz_attn(save_dir, spatial_rollout_attn, temporal_rollout_attn, originals, preds, labels, names)
    



if __name__ == "__main__":
    args = arg_parser()
    main(args)