import torch


def match_name_keywords(n, name_keywords):
    out = False
    for b in name_keywords:
        if b in n:
            out = True
            break
    return out


def build_optimizer(args, model):
    param_dicts = [
            {"params": [p for n, p in model.named_parameters() if not match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
              "lr": args.learning_rate,
            },

            {"params": [p for n, p in model.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
              "lr": args.lr_backbone,
            },
        ]
   
    # Optimization
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    elif args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(param_dicts, lr=args.learning_rate, weight_decay=1e-4, eps=1e-8, betas=(0.9, 0.999))
    return optimizer


def build_lr_scheduler(args, optimizer):
    if args.lr_scdler == "default":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_epoch_size, gamma=args.scheduler_gamma)
    elif args.lr_scdler == 'multi_step':
        #milestones=args.sch
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 8, 11, 13, 15], gamma=0.5)
    elif args.lr_scdler == "lr_drop":  
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop, gamma=args.scheduler_gamma)
    else:
        raise NotImplementedError
        
    return lr_scheduler