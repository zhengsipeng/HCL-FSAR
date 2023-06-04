from torch.utils.data import DataLoader
from dataset.sampler import EpisodeSampler, SupConSampler
from dataset.data_reader import GeneralDataset


def prepare_dataset(args):

    train_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        frame_size=args.frame_size,
        setname=args.setname,
    )

    val_dataset = GeneralDataset(
        args,
        frames_path=args.frames_path,
        frame_size=args.frame_size,
        setname='test',
    )

    if args.meta_learn:
        print("total training episodes: {}".format(args.num_epochs * args.num_train_episode))
        train_sampler = EpisodeSampler(train_dataset.classes, args.num_train_episode, args.way, args.shot, args.query)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    elif args.method == "hcl":
        train_sampler = SupConSampler(train_dataset.classes, args.num_train_episode, args.batch_size)
        train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, sampler=None)

    print("[train] number of videos / classes: {} / {}, ".format(len(train_dataset), train_dataset.num_classes))
    print("[val] number of videos / classes: {} / {}".format(len(val_dataset), val_dataset.num_classes))
  
    val_sampler = EpisodeSampler(val_dataset.classes, args.num_val_episode, args.way, args.shot, args.query)
    val_loader = DataLoader(dataset=val_dataset, batch_sampler=val_sampler, num_workers=4, pin_memory=True)

    return train_loader, val_loader