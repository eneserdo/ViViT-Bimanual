import pickle
import random
from os.path import join as opj

import matplotlib.pyplot as pyplot
import matplotlib
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from models import make_model
from dataset import (ACTIONS, OBJECTS, RELATIONS, _get_max_length,
                     get_bimanual_dataset_splits, get_processing_dir)
from utils import (Metrics, VisIndividualPreds,
                   arg_parser, get_loss_weights, get_minutes, metrics_to_str,
                   seed_worker, visualize_incorrect_preds, wandb_logging)


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

pyplot.ioff()
matplotlib.use('Agg')

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)


def get_sample_batch(args):
    TRAIN_SUBSET = [150, 500, 1500]
    VAL_SUBSET = [50, 100, 200]
    
    train_set, test_set, val_set = get_bimanual_dataset_splits(root=args.root, 
                                                            process_it_anyway=args.process_it,
                                                            test_subject=args.test_subject,
                                                            validation_take=args.validation_take,
                                                            temporal_length=args.temporal_length,
                                                            downsample=args.downsample,
                                                            use_vf=args.use_vf,
                                                            monograph=args.monograph)

    train_samples = []
    for i in TRAIN_SUBSET:
        train_samples.append(train_set[i])
    
    val_samples = []
    for i in VAL_SUBSET:
        val_samples.append(val_set[i])
    
    with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
        sample_dict = pickle.load(f)
    
    # sample_dict["train"]
    labels = []
    names = []


    return torch.stack(train_samples + val_samples), labels, names


ATTENTION = {}
def getAttentionHook(name):
    def hook(model, input, output):
        # model is supposed to be MultiHeadAttention
        # assuming batch_first=False

        x = input[0].detach() # y = mha(x, x, x)
        num_heads = model.num_heads
        head_dim = model.head_dim
        seq_len, bs, embed_dim = x.shape

        wq, wk, wv = torch.split(model.in_proj_weight, [embed_dim, embed_dim, embed_dim], dim=0)   
        bq, bk, bv = torch.split(model.in_proj_bias, [embed_dim, embed_dim, embed_dim], dim=0)

        q = torch.matmul(x, wq.T) + bq
        k = torch.matmul(x, wk.T) + bk
        # v = torch.matmul(x, wv.T) + bv

        q = q.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)
        k = k.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)
        # v = v.reshape(seq_len, num_heads*bs, head_dim).transpose(0, 1)

        scale = 1.0 / np.sqrt(head_dim)

        attn=torch.bmm(q, k.transpose(1, 2))
        attn=torch.softmax(attn*scale, dim=-1)

        attn_reshaped = attn.reshape(bs, num_heads, seq_len, seq_len)
        attn_reshaped_avr = attn_reshaped.mean(dim=1)
        attn_reshaped_max = attn_reshaped_avr.max(dim=1).values

        # double check
        # y = torch.bmm(attn, v)
        # y = y.transpose(0, 1).reshape(seq_len*bs, embed_dim)
        # y = y @ model.out_proj.weight.T + model.out_proj.bias
        # y = y.reshape(seq_len, bs, embed_dim)
        # assert torch.allclose(output[0], y, atol=1e-4)
        # what about the bias and dropout?
        ### 

        ATTENTION[name+"_avr"] = attn_reshaped_avr
        ATTENTION[name+"_max"] = attn_reshaped_max
        
    return hook

def set_hooks(model):
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

    device = torch.device('cuda:' + str(args.gpu_id) if torch.cuda.is_available() else 'cpu')


    # load model
    args.return_attention = True

    model = make_model(args).to(device)
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])


    # save transformer attentions via forward hook

    # load data

    samples, labels = get_sample_batch(args)    


    with torch.no_grad():
        # load model
        pred = model(samples.to(device))
        pred = pred.cpu().numpy()
        labels = labels.cpu().numpy()
            
            


if __name__ == "__main__":
    args = arg_parser()
    main(args)