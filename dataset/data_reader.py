import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from dataset.transformer import ImageNetPolicy, Lighting
import pdb


class BaseDataset(Dataset):
    def __init__(self, args, frames_path, frame_size, setname='train', 
                 random_pad_sample=False, uniform_frame_sample=False, random_start_position=False,
                 max_interval=64, random_interval=False):
        self.args = args
        self.vis = args.vis
        self.data_aug = args.data_aug
        self.dataset = args.dataset
        self.sequence_length = args.sequence_length
        self.setname = setname
        self.reverse = args.reverse
        self.use_stack = False
        if self.args.meta_learn or ('train' not in self.setname):
            self.use_stack = True

        # pad option => using for _add_pads function
        self.random_pad_sample = random_pad_sample

        # frame sampler option => using for _frame_sampler function
        self.uniform_frame_sample = uniform_frame_sample
        self.random_start_position = random_start_position
        self.max_interval = max_interval
        self.random_interval = random_interval
        
        assert setname in ['train', 'test', 'val', 'train_val'], "'{}' is not valid setname.".format(setname)

        setnames = setname.split('_')
        class_folders = []
        for setname in setnames:
            with open(args.class_split_folder+'%s_class.txt'%setname, 'r') as f:
                class_folders += [clss.strip() for clss in f.readlines()]
        
        class_folders.sort()
        self.class_folders = class_folders
        self.num_classes = len(class_folders)
    
        # this value will using for CategoriesSampler class
        self.classes = [] 
        self.data_paths = []
        self.cls_name2id = {}
        self.cls_id2name = {}
     
        for cls_id, class_folder in enumerate(self.class_folders):
            video_ids = os.listdir(os.path.join(frames_path, class_folder))
            video_ids.sort()
            self.cls_name2id[class_folder] = cls_id
            self.cls_id2name[cls_id] = class_folder
            for video_id in video_ids:
                self.data_paths.append(os.path.join(frames_path, class_folder, video_id))
                self.classes.append(cls_id)
        
        if self.reverse:
            rv_root = "data/images/reverse_images/%s"%self.dataset
            rv_class_folders = os.listdir(rv_root)
            for rv_cls_folder in rv_class_folders:
                if rv_cls_folder not in self.class_folders:
                    self.class_folders.append(rv_cls_folder)
                    cls_id = len(self.class_folders)-1
                    self.cls_name2id[rv_cls_folder] = cls_id
                    self.cls_id2name[cls_id] = rv_cls_folder
                else:
                    for i, class_folder in enumerate(self.class_folders):
                        if rv_cls_folder == class_folder:
                            cls_id = i
                
                video_ids = os.listdir(rv_root+"/"+rv_cls_folder)
                video_ids.sort()
                for video_id in video_ids:
                    self.data_paths.append(rv_root+"/"+rv_cls_folder+"/"+video_id)
                    self.classes.append(cls_id)
   
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.transform = transforms.Compose([
                transforms.Resize((frame_size, frame_size)),
                transforms.ToTensor(),
                normalize,
            ])

        if "train" in self.setname:
            
            if self.data_aug == "default":
                self.transform = transforms.Compose([
                    transforms.Resize((frame_size, frame_size)),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif self.data_aug == "type1":
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=frame_size, scale=(0.2, 1.)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([
                        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif self.data_aug == "type2":
                self.transform = transforms.Compose([
                    transforms.Resize((frame_size + 16, frame_size + 59)),
                    transforms.CenterCrop((frame_size, frame_size)),
                    
                    transforms.ColorJitter(
                        brightness=0.4, contrast=0.4, saturation=0.4
                    ),
                    transforms.ToTensor(),
                    Lighting(alphastd=0.1, eigval=[0.2175, 0.0188, 0.0045],
                                            eigvec=[[-0.5675, 0.7192, 0.4009],
                                                    [-0.5808, -0.0045, -0.8140],
                                                    [-0.5836, -0.6948, 0.4203]]
                    ),
                    normalize,
                ])
            elif self.data_aug == "auto":
                # autoaugment transformer for insufficient frames in training phase
                self.transform = transforms.Compose([
                    transforms.Resize((frame_size + 16, frame_size + 59)),
                    transforms.CenterCrop((frame_size, frame_size)),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize,
                ])
            elif self.data_aug == "auto2":
                # autoaugment transformer for insufficient frames in training phase
                self.transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=frame_size, scale=(0.2, 1.)),
                    ImageNetPolicy(),
                    transforms.ToTensor(),
                    normalize,
                ])

    def add_pads_sequence(self, sorted_frames_path):
        # get sorted frames length to list
        sequence = np.arange(len(sorted_frames_path))
        
        if self.random_pad_sample:
            # random sampling of pad
            add_sequence = np.random.choice(sequence, self.sequence_length - len(sequence))
        else:
            # repeated of first pad
            add_sequence = np.repeat(sequence[0], self.sequence_length - len(sequence))
        
        # sorting the pads
        sequence = sorted(np.append(sequence, add_sequence, axis=0))
        
        return sequence

    def sequence_sampler(self, sorted_frames_path):
        # get sorted frames length to list
        sorted_frames_length = len(sorted_frames_path)

        # set a sampling strategy
        if self.uniform_frame_sample:
            # set a default interval
            interval = (sorted_frames_length // self.sequence_length) - 1
            if self.max_interval != -1 and interval > self.max_interval:
                interval = self.max_interval

            # set a interval with randomly
            if self.random_interval:
                interval = np.random.permutation(np.arange(start=0, stop=interval + 1))[0]
    
            # get a require frames
            require_frames = ((interval + 1) * self.sequence_length - interval)
            
            # get a range of start position
            range_of_start_position = sorted_frames_length - require_frames

            # set a start position
            if self.random_start_position:
                start_position = np.random.randint(0, range_of_start_position + 1)
            else:
                start_position = 0
            
            sequence = list(range(start_position, require_frames + start_position, interval + 1))
        else:
            sequence = sorted(np.random.permutation(np.arange(sorted_frames_length))[:self.sequence_length])

        return sequence
    
    def get_iter_data(self, i):
        # get frames and sort
        data_path = self.data_paths[i]
        
        vid = data_path.split("/")[-1]
        clss = data_path.split("/")[-2]
        label = self.classes[i]

        sorted_frames_path = sorted(glob.glob(data_path+"/*.jpg"), key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
        sorted_frames_length = len(sorted_frames_path)
        if sorted_frames_length == 0:
            imgs = [data_path+"/"+img for img in os.listdir(data_path)]
            sorted_frames_path = sorted(imgs, key=lambda path: int(path.split(".jpg")[0].split("\\" if os.name == 'nt' else "/")[-1]))
            sorted_frames_length = len(sorted_frames_path)
        assert sorted_frames_length != 0, "'{}' Path is not exist or no frames in there.".format(data_path)

        # we may be encounter that error such as
        # 1. when insufficient frames of video rather than setted sequence length, _add_pads function will be solve this problem
        # 2. when video has too many frames rather than setted sequence length, _frame_sampler function will be solve this problem
        if sorted_frames_length < self.sequence_length:
            datas, seq_ids, frames = self._add_pads(sorted_frames_path)
        else:
            datas, seq_ids, frames = self._frame_sampler(sorted_frames_path)
        
        datas = torch.stack(datas)

        if "train" not in self.setname and self.vis:
            frames = torch.stack(frames)
        else:
            frames = label

        return datas, vid, clss, label, frames


# ==================
# General Datastet
# ==================
class GeneralDataset(BaseDataset):
    def __init__(self, args, frames_path, frame_size, setname='train',
                 random_pad_sample=False, uniform_frame_sample=False, random_start_position=False,
                 max_interval=64, random_interval=False):
        super(GeneralDataset, self).__init__(args, frames_path, frame_size, setname, random_pad_sample, 
                            uniform_frame_sample, random_start_position, max_interval, random_interval)
        
        self.frame_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
 
    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        sequence = self.add_pads_sequence(sorted_frames_path)

        # transform to Tensor
        frames = [Image.open(sorted_frames_path[s]) for s in sequence]
        datas = [self.transform(frame) for frame in frames]
        
        if "train" not in self.setname and self.vis:
            frames = [self.frame_transform(frame) for frame in frames]
        else:
            frames = None
        #datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]

        return datas, sequence, frames

    def _frame_sampler(self, sorted_frames_path):
        sequence = self.sequence_sampler(sorted_frames_path)

        # transform to Tensor
        frames = [Image.open(sorted_frames_path[s]) for s in sequence]
        datas = [self.transform(frame) for frame in frames]

        if "train" not in self.setname and self.vis:
            frames = [self.frame_transform(frame) for frame in frames]
        else:
            frames = None
        #datas = [self.transform(Image.open(sorted_frames_path[s])) for s in sequence]
        
        return datas, sequence, frames

    def __getitem__(self, index):
        datas, vid, clss, label, frames = self.get_iter_data(index % len(self))

        return datas, vid, label 


# ==================
# Contrast Datastet
# ==================
class ContrastDataset(BaseDataset):
    def __init__(self, args, frames_path, frame_size, setname='train',
                random_pad_sample=False, uniform_frame_sample=False, random_start_position=False,
                 max_interval=64, random_interval=False):
        super(ContrastDataset, self).__init__(args, frames_path, frame_size, setname, random_pad_sample, 
                            uniform_frame_sample, random_start_position, max_interval, random_interval)

    def __len__(self):
        return len(self.data_paths)

    def _add_pads(self, sorted_frames_path):
        sequence = self.add_pads_sequence(sorted_frames_path)

        # transform to Tensor
        datas = []
        imgs = [Image.open(sorted_frames_path[s]) for s in sequence]

        _datas = [self.transform(imgs[s]) for s in range(len(sequence))]
        datas.append(torch.stack(_datas))

        _datas = [self.transform(imgs[s]) for s in range(len(sequence))]
        datas.append(torch.stack(_datas))

        return datas, sequence, None

    def _frame_sampler(self, sorted_frames_path):
        sequence = self.sequence_sampler(sorted_frames_path)

        # transform to Tensor
        datas = []
        imgs = [Image.open(sorted_frames_path[s]) for s in sequence]

        _datas = [self.transform(imgs[s]) for s in range(len(sequence))]
        datas.append(torch.stack(_datas))

        _datas = [self.transform(imgs[s]) for s in range(len(sequence))]
        datas.append(torch.stack(_datas))

        return datas, sequence, None

    def __getitem__(self, index):
        datas, _, modal_aux, label, _ = self.get_iter_data(index % len(self))
        
        return datas, label

