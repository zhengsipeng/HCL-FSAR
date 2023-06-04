import argparse
import pdb
from .misc import write_logs


def init_args():
    parser = argparse.ArgumentParser()
    # ===========================Basic options===================================
    parser.add_argument("--method", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")
    parser.add_argument("--dataset", type=str, default="ssv2_100_otam")
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--print_iter", type=int, default=10)
    parser.add_argument("--save_iter", type=int, default=500)
    parser.add_argument("--ckpt_path", type=str, default="")

    # ===========================Few-shot options=================================
    parser.add_argument("--meta_learn", action="store_true")
    parser.add_argument("--way", type=int, default=5)
    parser.add_argument("--shot", type=int, default=1)
    parser.add_argument("--query", type=int, default=5)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_train_episode", type=int, default=200)
    parser.add_argument("--num_val_episode", type=int, default=3000)
    parser.add_argument("--episode_per_batch", type=int, default=16)
    parser.add_argument("--sim_metric", type=str, default='cosine')  # LR, NN, Cosine, Proto, SVM

    # ===========================Dataset options==================================
    parser.add_argument("--class_split_folder", type=str, default="data/splits/")
    parser.add_argument("--frames_path", type=str, default="data/images/")
    parser.add_argument("--labels_path", type=str, default="data/splits/")
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--data_aug", type=str, default='default')
    parser.add_argument("--setname", type=str, default="train_val")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--val_data", type=str, default='')  # for cross domain
    parser.add_argument("--amp_opt_level", type=str, default="O1", choices=["O0", "O1", "O2"],
                        help="mixed precision opt level, if O0, no amp is used")
    parser.add_argument("--frame_size", type=int, default=224)
    parser.add_argument("--sequence_length", type=int, default=8)
    parser.add_argument("--random_pad_sample", action="store_true")
    parser.add_argument("--pad_option", type=str, default="default")
    parser.add_argument("--uniform_frame_sample", action="store_true")
    parser.add_argument("--random_start_position", action="store_true")
    parser.add_argument("--max_interval", type=int, default=64)
    parser.add_argument("--random_interval", action="store_true")

    # ===========================Backbone options=================================
    parser.add_argument("--backbone", type=str, default="resnet50")  # resnet18, resnet34, resnet50
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")

    # ===========================Transformer options===================
    parser.add_argument('--transformer_name', default="vanilla", type=str)
    parser.add_argument('--enc_layers', default=1, type=int, help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=2, type=int, help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int, help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--d_model', default=256, type=int, help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--tf_dropout', default=0.1, type=float, help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int, help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--late_fc', action='store_true')
    parser.add_argument('--dilation', action='store_true', help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--num_group', default=16, type=int)
    parser.add_argument('--position_embedding', default='3d_learned', type=str, choices=('2d_sine', '2d_learned', '3d_learned'),
                        help="Type of positional embedding to use on top of the image features")
    
    # ===========================Key training options==============================
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--lr_scdler", type=str, default="default")
    parser.add_argument("--lr_drop", type=int, default=5)
    parser.add_argument("--scheduler_epoch_size", type=int, default=4)
    parser.add_argument("--scheduler_gamma", type=float, default=0.5)
    parser.add_argument("--bn_threshold", type=float, default=2e-2)   
    parser.add_argument("--optimizer", type=str, default='sgd')
    parser.add_argument('--lr_backbone_names', default=["encoder.module"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--clip_max_norm', default=0.1, type=float)
    
    # ===========================Contrastive options========================
    parser.add_argument("--temp", type=float, default=0.07)
    parser.add_argument("--use_spatial", action="store_true")
    parser.add_argument("--use_spa_cycle", action="store_true")
    parser.add_argument("--use_spa_mscale", action="store_true")
    parser.add_argument("--use_semantic", action="store_true")
    parser.add_argument('--temp_set', default=[2], type=int, nargs='+')
    parser.add_argument("--sigma_ce", type=float, default=1.0)
    parser.add_argument("--sigma_global", type=float, default=1.0)
    parser.add_argument("--sigma_temp", type=float, default=1.0)
    parser.add_argument("--sigma_spa", type=float, default=1.0)
    parser.add_argument("--sigma_temp_cycle", type=float, default=0.25)
    parser.add_argument("--sigma_spa_cycle", type=float, default=0.25)
    parser.add_argument("--topT", type=int, default=10)
    parser.add_argument("--topK", type=int, default=40)
    parser.add_argument("--bert_finetune", action="store_true")
    args = parser.parse_args()

    args.class_split_folder = args.class_split_folder + args.dataset + "/"
    args.frames_path = args.frames_path + args.dataset + "/"
    args.labels_path = args.labels_path + args.dataset + "/"
    args.lr_backbone = args.learning_rate / 10
    
    if args.shot == 1:
        args.sim_metric = "cosine"
    else:
        args.sim_metric = "Proto"

    if args.model_name == "":
        args.model_name = "%s_%s_bsz%d_layer%d_%d_%s_%.4f_aug-%s_temp-%.2f_ce-%.2f_cts-%.2f"%(
                    args.dataset, args.method, args.batch_size,
                    args.enc_layers, args.d_model, args.optimizer, args.learning_rate, 
                    args.data_aug, args.temp, args.sigma_ce, args.sigma_global)        

    args.save_path = "models/%s"%args.model_name
    write_logs(args)
    
    return args