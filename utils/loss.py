import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchCycleLoss(nn.Module):
    def __init__(self, args):
        super(PatchCycleLoss, self).__init__()
        self.sigma_ce = args.sigma_ce
        self.sigma_sp = args.sigma_sp
        self.sigma_feat = args.sigma_feat
        self.criterion_ce = nn.CrossEntropyLoss(reduction="mean")
        self.criterion_mse = nn.MSELoss()

    def forward(self, logits, labels, st_locs, st_locs_back, p_sim_12, p_sim_21, pos_onehot):
        """
        pos_onehot: batchsize, p_num
        p_sim_12 & p_sim_21, batchsize, p_num
        """
        p_num = pos_onehot.shape[1]
        pos_num = pos_onehot.sum(1)  # b, 1
    
        # cross entropy loss for action classification
        ce_loss = self.criterion_ce(logits, labels)*self.sigma_ce

        # MSE loss for spatial coordinates
        sp_dist = torch.sqrt(((st_locs - st_locs_back)**2).sum(2))  # b, p_num

        log_softmax_sp = -pos_onehot * torch.log(F.softmax(sp_dist, dim=1))  # b, p_num
        infonce_sp_loss = torch.div(log_softmax_sp.sum(1), pos_num+1e-4).mean()*self.sigma_sp

        # InfoNCE loss for contrastive leraning
        # type1
        prob = p_sim_12 * p_sim_21*10  # b, p_num
        #print(prob)
        prob = F.softmax(prob, dim=1)
        #print(prob[0])
        # normalization
        mean = prob.mean(dim=1).unsqueeze(1)
        stdv = torch.sqrt(((prob-mean)**2).sum(dim=1)/(p_num-1)).unsqueeze(1)
        prob = ((prob-mean)/stdv)
        softmax_prob = F.softmax(prob, dim=1)
        log_softmax = -pos_onehot * torch.log(softmax_prob)

        info_nce_loss = torch.div(log_softmax.sum(1), pos_num+1e-4).mean()*self.sigma_feat
        
        losses = ce_loss + infonce_sp_loss + info_nce_loss

        return losses, ce_loss, infonce_sp_loss, info_nce_loss


def cross_entropy_loss(test_logits_sample, test_labels):
    return F.cross_entropy(test_logits_sample, test_labels)


def trx_cross_entropy_loss(test_logits_sample, test_labels):
    """
    Compute the classification loss.
    """
    # test_logits_sample: [1, way * num_query, way]
    # test_labels: [way * num_query]
   
    test_logits_sample = test_logits_sample.unsqueeze(0)
    size = test_logits_sample.size()
    sample_count = size[0]  # scalar for the loop counter
    num_samples = torch.tensor([sample_count], dtype=torch.float, requires_grad=False).cuda()
    
    log_py = torch.empty(size=(size[0], size[1]), dtype=torch.float).cuda()
    for sample in range(sample_count):
        log_py[sample] = -F.cross_entropy(test_logits_sample[sample], test_labels, reduction='none')

    score = torch.logsumexp(log_py, dim=0) - torch.log(num_samples)

    return -torch.sum(score, dim=0)


class SupConLoss(nn.Module):
    def __init__(self, args, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.args = args
        self.use_ce = args.use_ce
        self.use_contrast = args.use_contrast
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.sigma_cts = args.sigma_cts
        self.sigma_ce = args.sigma_ce

    def forward(self, features, preds, labels, loss_type=''):
        """
        features need to be [batchsize, n_views, ...] at least 3 dimenstions are required
        """
       
        device = torch.device('cuda')
        batch_size = features.shape[0]
   
        if loss_type == 'SimCLR':
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif loss_type == 'SupCon':
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)  # bsz, bsz
        
        contrast_count = features.shape[1]  # n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # bsz*2, c
        
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count  # 2
        
        # compute logits
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()  # bsz*2, bsz*2

        
        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )  # mask i=i
        mask = mask * logits_mask
 
        # compute log prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))  # log(x/y) = logx - logy

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss 
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        contrast_loss = loss.view(anchor_count, batch_size).mean()
        
        if self.use_ce:
            labels = labels.reshape(-1)
            ce_loss = self.cross_entropy(preds, labels)  #*self.sigma_ce
        
        if self.use_ce and self.use_contrast:
            loss = self.sigma_cts + self.sigma_ce*ce_loss
            return loss, ce_loss, contrast_loss
        elif self.use_contrast and not self.use_ce:
            return contrast_loss, contrast_loss, contrast_loss
        elif not self.use_contrast and self.use_ce:   
            return ce_loss, ce_loss, ce_loss
            
        
class HCLLoss(nn.Module):
    def __init__(self, args, contrast_mode='all', base_temperature=0.07):
        super(HCLLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.temperature = args.temp
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.sigma_global = args.sigma_global
        self.sigma_temp = args.sigma_temp
        self.sigma_spa = args.sigma_spa
        self.sigma_ce = args.sigma_ce
        self.sigma_temp_cycle = args.sigma_temp_cycle
        self.sigma_spa_cycle = args.sigma_spa_cycle

    def forward(self, contrasts, preds, labels):
        """
        features need to be [batchsize, n_views, ...] at least 3 dimenstions are required
        """

        labels = labels.reshape(-1)

        ce_loss = self.sigma_ce * self.cross_entropy(preds, labels)  #*self.sigma_ce
        contrasts["global"] = self.sigma_global * contrasts["global"]
        contrasts["temp"] = self.sigma_temp * contrasts["temp"] 
        contrasts["temp_cycle"] = self.sigma_temp_cycle * contrasts["temp_cycle"]
        contrasts["spatial"] = self.sigma_spa * contrasts["spatial"] 
        contrasts["spa_cycle"] = self.sigma_spa_cycle * contrasts["spa_cycle"]
                               
        contrast_loss = contrasts["global"] + contrasts["temp"] + contrasts["spatial"] + \
                        contrasts["temp_cycle"] + contrasts["spa_cycle"]
        loss = ce_loss + contrast_loss
        
        return loss, ce_loss, contrasts