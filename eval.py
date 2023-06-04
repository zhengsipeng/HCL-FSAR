import torch
from dataset.prepare_dataset import prepare_dataset
from nets.hcl import HCL
from nets.meta_nets import MetaNets
from utils.misc import precedding, printer
from utils.optim import build_optimizer
from utils.configs import init_args
import warnings
warnings.filterwarnings('ignore')
try:
    from apex import amp
except ImportError:
    amp = None


def eval(args):
    _, val_loader = prepare_dataset(args)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.method == "hcl":
        model = HCL(args)
    else:
        model = MetaNets(args)

    optimizer = build_optimizer(args, model)
    model = model.to(device)
    if args.num_gpus > 1:
        model.distribute_model(args.num_gpus)

    checkpoint = torch.load(args.ckpt_path)
    for k, v in checkpoint.items():
        print(k)
    model.load_state_dict(checkpoint, strict=False)

    if args.amp_opt_level != "O0":
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp_opt_level)

    print("\nTest {}-way {}-shot {}-query".format(args.way, args.shot, args.query))
    print(args.ckpt_path)

    val_acc = []
    model.eval()

    with torch.no_grad():
        for i, (imgs, vids, labels) in enumerate(val_loader):
            support_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.shot).to(device).reshape(-1)
            query_labels = torch.arange(args.way).reshape(args.way, 1).repeat(1, args.query).to(device).reshape(-1)
            labels = torch.cat([support_labels, query_labels])

            imgs = imgs.to(device)  # way*shot+way*query
            
            if args.method == "hcl":
                pred = model.test(imgs, vids)
            else:
                logits, pred = model(imgs, labels[args.way*args.shot:])
                    
            acc = (pred.view(-1) == query_labels).type(torch.cuda.FloatTensor).mean().item()
    
            val_acc.append(acc)
            total_acc = sum(val_acc) / len(val_acc)
            
            printer("val", 0, args.num_epochs, i+1, len(val_loader), 0, 0, \
                    acc * 100, total_acc * 100, max_acc=0*100)


if __name__ == "__main__":
    args = init_args()  
    precedding(args)
    eval(args)
    