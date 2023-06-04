import os
import sys
import torch
import scipy.stats as stats
import numpy as np



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        
def printer_cycle(status, epoch, num_epochs, batch, num_batchs, loss, loss_mean, ce, ce_mean, 
            infonce_sp, infonce_sp_mean, infonce, infonce_mean, acc, acc_mean, pos_num, sim_thresh):
    sys.stdout.write("\r[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.4f} (mean: {:.4f}), Acc: {:.2f}% (mean: {:.2f}%)] [CE: {:.3f} (mean: {:.3f}), SP: {:.3f} (mean: {:.3f}), FEAT: {:.3f} (mean: {:.3f}), pos: {:.1f}, thresh: {:.4f}]".format(
            status, epoch, num_epochs, batch, num_batchs, loss, loss_mean, acc, acc_mean,
            ce, ce_mean, infonce_sp, infonce_sp_mean, infonce, infonce_mean, pos_num, sim_thresh,
        )
    )

    
def printer(status, epoch, num_epochs, batch, num_batchs, loss, loss_mean, acc, acc_mean, max_acc=None):
    if max_acc is None:
        sys.stdout.write("\r[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.4f} (mean: {:.4f}), Acc: {:.2f}% (mean: {:.2f}%)] ".format(
                status,
                epoch,
                num_epochs,
                batch,
                num_batchs,
                loss,
                loss_mean,
                acc,
                acc_mean
            )
        )
    else:
        sys.stdout.write("\r[{}]-[Epoch {}/{}] [Batch {}/{}] [Loss: {:.4f} (mean: {:.4f}), Acc: {:.2f}% (mean: {:.2f}%), Max_Acc: {:.4f}] ".format(
                status,
                epoch,
                num_epochs,
                batch,
                num_batchs,
                loss,
                loss_mean,
                acc,
                acc_mean,
                max_acc
            )
        )


def precedding(args):
    # path to save
    path_check(args.save_path)
    # path to tensorboard
    # print args and save it in the save_path
    args_print_save(args)


def path_check(path):
    if os.path.exists(path):
        response = 'y'
        '''
        while True:
            print("'{}' path is already exist, do you want continue after remove ? [y/n]".format(path))
            response = input()
            if response == 'y' or response == 'n':
                break
        '''
        #if response == 'y':
        #    shutil.rmtree(path)
        #    os.makedirs(path)

        if response == 'n':
            print("this script was terminated by user")
            sys.exit()
    else:
        os.makedirs(path)

def args_print_save(args):
    # print
    print("=================================================")
    [print("{}:{}".format(arg, getattr(args, arg))) for arg in vars(args)]
    print("=================================================")

    # save
    with open(os.path.join(args.save_path, "args.txt"), "w") as f:
        f.write("=================================================\n")
        [f.write("{}:{}\n".format(arg, getattr(args, arg))) for arg in vars(args)]
        f.write("=================================================\n")


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h # m +-h


def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')
    

def verify_checkpoint_dir(checkpoint_dir, resume, test_mode):
    if resume:  # verify that the checkpoint directory and file exists
        if not os.path.exists(checkpoint_dir):
            print("Can't resume for checkpoint. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
            sys.exit()

        checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.pt')
        if not os.path.isfile(checkpoint_file):
            print("Can't resume for checkpoint. Checkpoint file ({}) does not exist.".format(checkpoint_file), flush=True)
            sys.exit()
    #elif test_mode:
    #    if not os.path.exists(checkpoint_dir):
    #        print("Can't test. Checkpoint directory ({}) does not exist.".format(checkpoint_dir), flush=True)
    #        sys.exit()
    else:
        print('~~~')
        '''
        if os.path.exists(checkpoint_dir):
            print("Checkpoint directory ({}) already exits.".format(checkpoint_dir), flush=True)
            print("If starting a new training run, specify a directory that does not already exist.", flush=True)
            print("If you want to resume a training run, specify the -r option on the command line.", flush=True)
            sys.exit()
        '''


def sample_normal(mean, var, num_samples):
    """
    Generate samples from a reparameterized normal distribution
    :param mean: tensor - mean parameter of the distribution
    :param var: tensor - variance of the distribution
    :param num_samples: np scalar - number of samples to generate
    :return: tensor - samples from distribution of size numSamples x dim(mean)
    """
    sample_shape = [num_samples] + len(mean.size())*[1]
    normal_distribution = torch.distributions.Normal(mean.repeat(sample_shape), var.repeat(sample_shape))
    return normal_distribution.rsample()


def aggregate_accuracy(test_logits_sample, test_labels):
    """
    Compute classification accuracy.
    """
    #print(test_logits_sample)
    averaged_predictions = torch.logsumexp(test_logits_sample, dim=0)
    #print(averaged_predictions)
    return torch.mean(torch.eq(test_labels, torch.argmax(averaged_predictions, dim=-1)).float())


def task_confusion(test_logits, test_labels, real_test_labels, batch_class_list):
    print(test_logits.shape)
    assert 1==0
    preds = torch.argmax(torch.logsumexp(test_logits, dim=0), dim=-1)
    real_preds = batch_class_list[preds]
    return real_preds


class TestAccuracies:
    """
    Determines if an evaluation on the validation set is better than the best so far.
    In particular, this handles the case for meta-dataset where we validate on multiple datasets 
    and we deem the evaluation to be better if more than half of the validation accuracies 
    on the individual validation datsets are better than the previous best.
    """

    def __init__(self, validation_datasets):
        self.datasets = validation_datasets
        self.dataset_count = len(self.datasets)
#        self.current_best_accuracy_dict = {}
#        for dataset in self.datasets:
#            self.current_best_accuracy_dict[dataset] = {"accuracy": 0.0, "confidence": 0.0}

#    def is_better(self, accuracies_dict):
#        is_better = False
#        is_better_count = 0
#        for i, dataset in enumerate(self.datasets):
#            if accuracies_dict[dataset]["accuracy"] > self.current_best_accuracy_dict[dataset]["accuracy"]:
#                is_better_count += 1
#
#        if is_better_count >= int(math.ceil(self.dataset_count / 2.0)):
#            is_better = True
#
#        return is_better

#    def replace(self, accuracies_dict):
#        self.current_best_accuracy_dict = accuracies_dict

    def print(self, logfile, accuracy_dict):
        print_and_log(logfile, "")  # add a blank line
        print_and_log(logfile, "Test Accuracies:")
        for dataset in self.datasets:
            print_and_log(logfile, "{0:}: {1:.1f}+/-{2:.1f}".format(dataset, accuracy_dict[dataset]["accuracy"],
                                                                    accuracy_dict[dataset]["confidence"]))
        print_and_log(logfile, "")  # add a blank line



# =======
# logs
# =======
def print_and_log(log_file, message):
    """
    Helper function to print to the screen and the cnaps_layer_log.txt file.
    """
    print(message, flush=True)
    log_file.write(message + '\n')


def get_log_files(checkpoint_dir, resume, test_mode):
    """
    Function that takes a path to a checkpoint directory and returns a reference to a logfile and paths to the
    fully trained model and the model with the best validation score.
    """
    verify_checkpoint_dir(checkpoint_dir, resume, test_mode)
    #if not test_mode and not resume:
    if not resume and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path_validation = os.path.join(checkpoint_dir, 'best_validation.pt')
    checkpoint_path_final = os.path.join(checkpoint_dir, 'fully_trained.pt')
    logfile_path = os.path.join(checkpoint_dir, 'log.txt')
    if os.path.isfile(logfile_path):
        logfile = open(logfile_path, "a", buffering=1)
    else:
        logfile = open(logfile_path, "w", buffering=1)

    return checkpoint_dir, logfile, checkpoint_path_validation, checkpoint_path_final


def write_logs(args):
    log_dict = {"method": args.method, "dataset":args.dataset, 
                "reverse": args.reverse, "batch_size": args.batch_size,
                "num_enc_layers": args.enc_layers, "d_model": args.d_model, 
                "optimizer": args.optimizer, "lr_rate": args.learning_rate, 
                "augmentation": args.data_aug, "temperature": args.temp, 
                "use_spa": args.use_spatial, "use_spa_cycle": args.use_spa_cycle,
                "sigma_ce": args.sigma_ce, "sigma_global": args.sigma_global,
                "sigma_temp": args.sigma_temp, "sigma_spa": args.sigma_spa}

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    with open("%s/configs.txt"%args.save_path, "w") as f:
        for k, v in log_dict.items():
            f.writelines("%s: %s\n"%(k, v))