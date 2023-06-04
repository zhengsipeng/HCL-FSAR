import os
import torch
import torch.nn as nn

from meta_trainer import meta_trainer
from dataset.prepare_dataset import prepare_dataset
from nets.hcl import HCL
from nets.meta_nets import MetaNets

from utils.loss import HCLLoss, cross_entropy_loss
from utils.misc import precedding, printer
from utils.configs import init_args
from utils.optim import build_optimizer, build_lr_scheduler
import warnings
warnings.filterwarnings('ignore')
try:
    from apex import amp
except ImportError:
    amp = None
import pdb


def meta_eval(e, args, val_loader, model, max_acc, best_epo):
    print("\nTest Epoch_{}... {}-way {}-shot {}-query".format(e, args.way, args.shot, args.query))
    print(args.model_name)
    val_acc = []
    model.eval()
    print("Best Epoch: %d"%best_epo)
    with torch.no_grad():
        for i, (imgs, vids, _) in enumerate(val_loader):
            support_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.shot).to(device).reshape(-1)
            query_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.query).to(device).reshape(-1)
            labels = torch.cat([support_labels, query_labels])

            imgs = imgs.to(device)  # way*shot+way*query
 
            if args.method == "hcl":
                pred = model.test(imgs, vids)
            else:
                _, pred = model(imgs, labels)
 
            acc = (pred.view(-1) == query_labels).type(torch.cuda.FloatTensor).mean().item()

            val_acc.append(acc)
            total_acc = sum(val_acc) / len(val_acc)
            printer("val", e, args.num_epochs, i+1, len(val_loader), 0, 0, acc * 100, total_acc * 100, max_acc=max_acc*100)
            #break
        if total_acc > max_acc:
            torch.save(model.state_dict(), os.path.join(args.save_path, "best.pth"))
            max_acc = total_acc
            best_epo = e

        with open("%s/logs.txt"%args.save_path, "a+") as f:
            f.writelines("Epoch %d: %f/%f\n"%(e, total_acc, max_acc))
        print("\n")

    return max_acc, best_epo


if __name__ == "__main__":
    args = init_args()  
    precedding(args)

    # Model Build 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method == "hcl":
        model = HCL(args)
        criterion = HCLLoss(args)
    else:
        model = MetaNets(args)
        criterion = nn.MSELoss() if args.method == "relation" else cross_entropy_loss #nn.CrossEntropyLoss()
    model = model.to(device)

    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)
    
    # Dataset
    train_loader, val_loader = prepare_dataset(args)

    # Optimizer and Lr Scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params: %.2fM'%(n_parameters/(1000*1000)))
    if hasattr(model, "flops"):
        flops = model.flop()
        print(f"number of GFLOPs: {flops / 1e9}")

    # Train and Eval
    max_acc, n_iter_train, best_epo = 0, 0, 0
    for e in range(args.num_epochs):
        lr = optimizer.param_groups[0]["lr"]
        tips = [e, n_iter_train, args.clip_max_norm, args.amp_opt_level, lr]
        n_iter_train = meta_trainer(args, model, criterion, optimizer, train_loader, device, tips)

        torch.save(model.state_dict(), os.path.join(args.save_path, "%d.pth"%e))
        lr_scheduler.step()

        max_acc, best_epo = meta_eval(e, args, val_loader, model, max_acc, best_epo)
            
        
