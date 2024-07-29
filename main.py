import glob
import os
import pickle
import random
import time
import warnings
from os.path import join as opj

import matplotlib.pyplot as pyplot
import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

import wandb
from dataset import (ACTIONS, get_bimanual_dataset_splits)
from model import make_model
from utils import (Metrics, VisIndividualPreds, arg_parser, get_loss_weights, get_minutes, metrics_to_str, seed_worker, wandb_logging)

# from torch_geometric.explain import Explainer, AttentionExplainer, ModelConfig



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True # probably disrupts reproducibility (among different machines), but it is faster

# # To fix the following error:
# OSError: [Errno 24] Too many open files
# Alternatively ulimit can be increased
# torch.multiprocessing.set_sharing_strategy('file_system')

# # An attempt to fix the following error:
# XIO:  fatal IO error 25 (Inappropriate ioctl for device) on X server "localhost:10.0" after 2627 requests (2627 known processed) with 8 events remaining.
pyplot.ioff()
matplotlib.use('Agg')

# To check for NaNs in the gradients
torch.autograd.set_detect_anomaly(True)


def run_one_epoch(model: torch.nn.Module, loader: DataLoader, mode: str, epoch: int):
    global step

    if mode in ["test", "val"]:
        model.eval()
    elif mode == "train":
        model.train()
    
    loss_cumulative = 0

    metrics = Metrics(args, len(ACTIONS), 32, device, gt_dict, sample_dict, mode)
    

    for imgs, labels, idx in loader:
        step += 1

        batch_size, n_frames = imgs.shape[0], imgs.shape[1]
        imgs = imgs.to(device)
        # imgs = imgs.half() # FIXME: half()
        labels = labels.to(device)
        # labels.shape: [batch_size, 32, 2]
        
        # bc ViViT processes tokens as 16x16x2 tubelets. So, temporal precision is halved during the training. 
        # But, to calculate the metrics, we will upsample the predictions to the original temporal resolution. 
        labels = labels[:, ::2, :] # FIXME: Hardcoded 2
        # labels.shape: [batch_size, 16, 2]
        
        labels = torch.cat([labels[:, :, 0], labels[:, :, 1]], dim=0)
        # labels.shape: [batch_size*2, 16] LLL...RRR...

        labels_flattened = labels.reshape(-1) #.flatten() 
        # labels_flattened.shape: [batch_size*2, 16] -> [batch_size*2*16]

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=not args.no_amp):
            pred = model(imgs)

            pred_flattened = pred.reshape(-1, pred.shape[-1]) #.flatten(0, 1)

            loss = criterion(pred_flattened, labels_flattened)
            loss /= args.num_acc
        
        loss_cumulative += loss.item()

        if mode == "train":
            scaler.scale(loss).backward()

            if (step % args.num_acc == 0): # or (step == len(loader))
                
                if args.clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        if mode != "test" and args.save_individual:
            # Visualize individiual predictions
            raise NotImplementedError("save_individual may be broken due to global idx, please check it")
            # VIP.pick_samples(y_true=gt_reordered.cpu().argmax(2), y_pred=pred.detach().cpu(), idx=idxes, epoch=epoch, mode=mode)


        step_f1_top1, step_f1_top3 = metrics.update(pred.detach(), labels, idx)

        num_of_log_per_epoch = 4
        if step % (len(loader)//num_of_log_per_epoch) == 0:
            print(f"{mode} E{epoch:03d} {step%len(loader):03d}/{len(loader)}: f1t1 {step_f1_top1:.4f} - f1t3 {step_f1_top3:.4f} - L {loss.item():.4f}", flush=True) 

        # if mode != "test":
        #     wandb.log({f"{mode}/batch": {'top1_F1': step_f1_top1, 'top3_F1': step_f1_top3, "Loss": loss.item()}, 'batch': step})

        # if mode == "test" and args.save_incorrects:
        #     raise NotImplementedError("save_incorrects may be broken, please check it")
        #     visualize_incorrect_preds(pred_top_1, ground_truth, idxes, num_of_samples=4, 
        #                                 test_idx_sample_mapping=dataset_mapping["test"]["sample"])
        # if (step+1)%20==0:
        #     print("20 BATCH ÇALIŞIYOR SADECE")
        #     break

        del imgs, labels, pred, pred_flattened, labels_flattened, loss

    # End of for loop
    
    metric_dict = metrics.compute()

    loss_cumulative = loss_cumulative / (len(loader.dataset) / args.batch_size)

    return metric_dict, loss_cumulative



def test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of):
    # Log everything for cross-validation 

    cv_file = opj(save_dir, f"cross_validation_fold{args.test_subject}.pickle")
    cv_list = []

    comparison_metric=choose_best_in_terms_of
    
    weight_list=sorted(glob.glob(opj(save_dir, "*.pt")))

    if len(weight_list) == 0:
        return None
    
    best_metric = 0    
    results = []
    best_i = -1
    for i, w_name in enumerate(weight_list):
        print(f"TESTING {w_name}:")
        checkpoint = torch.load(w_name)
        model.load_state_dict(checkpoint["model_state_dict"])
        returned_dict, _ = run_one_epoch(model, test_loader, 'test', epoch=int((w_name[-7:-3])))
        cv_list.append(returned_dict)
        
        print('### TEST: ', metrics_to_str(returned_dict))

        results.append(returned_dict)        
        if best_metric < returned_dict[comparison_metric]:
            best_metric = returned_dict[comparison_metric]
            best_i = i

    with open(cv_file, "wb") as f:
        pickle.dump(cv_list, f)

    print('### BEST: ', metrics_to_str(results[best_i]))
    return results[best_i]


if __name__ == "__main__":
    args = arg_parser()
    print("ARGS:\n", args)

    # save_dir = opj(os.getcwd(), "saved_models", f"{args.name}")
    save_dir = opj("saved_models", f"{args.name}")
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    # Make sure it is reproducible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.use_deterministic_algorithms(mode=True, warn_only=True) # This throws an error
    torch_generator = torch.Generator()
    torch_generator.manual_seed(args.seed)

    assert torch.cuda.is_available(), "No CUDA?"
    device = torch.device('cuda:' + str(args.gpu_id))

            
    wandb.init(project="Bimanual-Baseline", name=args.name, config=args.__dict__,
               mode="disabled" if args.disable_wandb else "online")


    train_set, test_set, val_set = get_bimanual_dataset_splits(args)


    train_loader = DataLoader(train_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)
    test_loader = DataLoader(test_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)
    val_loader = DataLoader(val_set, args.batch_size, num_workers=args.num_workers, shuffle=True, worker_init_fn=seed_worker, generator=torch_generator)

    training_len = len(train_set)
    val_len = len(val_set)
    print(f"Train: {training_len}, Val: {val_len}, Test: {len(test_set)}")

    model = make_model(args, device)
    # model = model.half() # FIXME: half()


    torch.save({"model_state_dict": model.state_dict()}, 
               opj(save_dir, f"model_epoch{0:0>4}.pt"))

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable params: {total_params:_}")

    wandb.watch(model, log='all')
    wandb.define_metric("epoch")
    wandb.define_metric("train/epoch*", step_metric="epoch")
    wandb.define_metric("val/epoch*", step_metric="epoch")
    wandb.define_metric("test/epoch*", step_metric="epoch")
    wandb.define_metric("val/confusion_matrix*", step_metric="epoch")
    wandb.define_metric("test/confusion_matrix*", step_metric="epoch")

    # if args.model == "tr":
    #     model_params = [
    #         {"params": model.spatial_model.parameters(), "lr": args.learning_rate * 0.1},
    #         {"params": model.temp_positional_encoding, "lr": args.learning_rate},
    #         {"params": model.temporal_model.parameters(), "lr": args.learning_rate},
    #         {"params": model.head.parameters(), "lr": args.learning_rate}
    #         # weight decay vs hepsi için ayrı ayrı yapılmalı mı?
    #     ]

    if args.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        print("args.momentum is ignored. It is only used for SGD.")
        optimizer = getattr(torch.optim, args.optimizer_type)(model.parameters(), lr=args.learning_rate,
                                                              weight_decay=args.weight_decay)


    if args.scheduler_step_size != -1:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step_size,
                                                    gamma=args.scheduler_gamma, verbose=True)
        
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.scheduler_gamma, patience=args.patience, verbose=True)


    # loss weights can be used to balance the classes using different strategies
    loss_weights = get_loss_weights(args.weighted_loss, None) # FIXME

    criterion = torch.nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.smoothing).to(device=device)

    scaler = torch.cuda.amp.GradScaler(enabled = not args.no_amp)


    
    # dataset_mapping contains mapping for each sample to its location in the videos. 
    # Also, a mapping from video to ground truth action labels
    with open(f"data/sample_dict_fold{args.test_subject}.pickle", "rb") as f:
        sample_dict = pickle.load(f)
    
    with open("data/gt_dict.pickle", "rb") as f:
        gt_dict = pickle.load(f)
    
    # split gt_dict into train, val, test
    gt_dict_train = {k: v for k, v in gt_dict.items() if k[0]!=str(args.test_subject) and k[2]!=str(args.validation_take)}
    gt_dict_test = {k: v for k, v in gt_dict.items() if k[0]==str(args.test_subject)}
    gt_dict_val = {k: v for k, v in gt_dict.items() if k[0]!=str(args.test_subject) and k[2]==str(args.validation_take)}
    gt_dict = {"train": gt_dict_train, "test": gt_dict_test, "val": gt_dict_val}


    if args.save_individual:
        raise NotImplementedError("save_individual may be broken, please check it")
        VIP = VisIndividualPreds(20, training_len, val_len, save_dir)



    # STARTS TRAINING AND VALIDATION
    step = 0
    print("Starts training...")

    for epoch in range(args.num_epochs):
        start = time.perf_counter()

        train_metrics, _ = run_one_epoch(model, train_loader, 'train', epoch)
        wandb_logging(wandb, train_metrics, epoch, "train")
        
        val_metrics, val_loss = run_one_epoch(model, val_loader, 'val', epoch)
        wandb_logging(wandb, val_metrics, epoch, "val")

        # Save model
        if epoch % args.saving_freq == 0 or epoch ==  args.num_epochs - 1:
            torch.save({"model_state_dict": model.state_dict(), 
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict()},
                        opj(save_dir, f"model_epoch{epoch:0>4}.pt"))


        scheduler.step(val_loss)
        stop = time.perf_counter()

        print(f'### {epoch:03d}: Time {get_minutes(stop - start)} m ### Train: ', metrics_to_str(train_metrics), ' ### Val: ', metrics_to_str(val_metrics))


    # save the individual predictions
    if args.save_individual:
        raise NotImplementedError("save_individual may be broken, please check it")
        VIP.save()

    # TEST
    print("Starts test...")

    choose_best_in_terms_of = "f1_top1_once"
    test_metrics = test_for_all_saved_weights(model, test_loader, choose_best_in_terms_of=choose_best_in_terms_of)
    
    if test_metrics is None:
        print("[WARN] No weights found to test")
    else:
        wandb_logging(wandb, test_metrics, epoch, "test")
    
    wandb.finish()
    
    # To prevent CUDA memory errors during consecutive runs
    try:
        torch.cuda.empty_cache()
    except RuntimeError as e:
        if "out of memory" in str(e):
            time.sleep(5)
        else:
            raise e