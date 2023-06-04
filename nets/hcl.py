from re import X
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle as pkl
from .base_net import BaseNet, Proto, Cosine, LR
from .encoder import SpatioTempEncoder
from torch.cuda.amp import autocast as autocast
from itertools import combinations 
from transformers.activations import gelu
import pdb


class HCL(BaseNet):
    """backbone + projection head"""
    def __init__(self, args, head='mlp', feat_dim=512):
        super(HCL, self).__init__(args)

        self.st_transformer_encoder = SpatioTempEncoder(args)
        self.temperature = args.temp        
        self.d_model = args.d_model
        self.num_group = args.num_group
        self.query_embed = nn.Parameter(torch.randn(1+self.num_group, args.d_model))  # zero is cls token

        self.action_classifier = nn.Linear(args.d_model, self.num_classes)
        self.bi_linear = nn.Linear(2*self.d_model, self.d_model)
        self.tri_linear = nn.Linear(3*self.d_model, self.d_model)
        self.temp_linear = nn.Linear(self.d_model, self.d_model)
        self.spa_linear = nn.Linear(self.d_model, self.d_model)
        self.norm_temp = nn.LayerNorm(self.d_model, eps=1e-12)
        self.norm_spa = nn.LayerNorm(self.d_model, eps=1e-12)

        self.use_spa_cycle = args.use_spa_cycle
        self.use_spa_mscale = args.use_spa_mscale
        self.use_semantic = args.use_semantic
        
        self.sigma_global = args.sigma_global
        self.sigma_temp = args.sigma_temp
        self.sigma_spa = args.sigma_spa

        self.topT = 10
        self.topK = 40
        
        self.sim_metric = args.sim_metric
        self.avgpool_2 = nn.AvgPool2d(2, 1)
        self.avgpool_4 = nn.AvgPool2d(4, 2)

        self.tuples = {}
        self.temp_set = args.temp_set
        for t in self.temp_set:
            frame_idxs = [i for i in range(self.seq_len)]
            frame_combinations = combinations(frame_idxs, t)
            self.tuples[t] = [torch.tensor(comb).cuda() for comb in frame_combinations]
        
        if self.use_semantic:
            self.bert_finetune = args.bert_finetune
            if self.bert_finetune:
                from transformers import BertTokenizer, BertModel
                with open("bert_model/action_lists/%s_vid2clsname.json"%self.dataset, "r") as f:
                    self.vid2clsname = json.load(f)
                self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
                self.bert = BertModel.from_pretrained('bert-base-cased')
            else:
                print("Pkl Loading")
                with open("bert_model/%s_vid2embid.json"%self.dataset, "r") as f:
                    self.vid2embid = json.load(f)
                with open("bert_model/%s_embs.pkl"%self.dataset, "rb") as f:
                    self.bert_embeddings = torch.as_tensor(pkl.load(f)).cuda()
                print("Pkl Loaded")
            
            self.emb_fc = nn.Linear(768, self.d_model)
            self.norm_emb = nn.LayerNorm(self.d_model, eps=1e-12)

    def get_global_contrast(self, hs):
        _hs = hs.mean(1).mean(1)
        
        fts = _hs.reshape(self.bsz, 2, -1) 
        fts1, fts2 = fts[:, 0, :], fts[:, 1, :]
        anchor_dot_contrast = torch.einsum("ij,ij->i", [fts1, fts2])
        anchor_dot_contrast = torch.div(anchor_dot_contrast, self.temperature)
        anchor_dot_contrast = anchor_dot_contrast.reshape(self.bsz, 1).repeat(1, 2).flatten() 
        exp_logits = torch.exp(anchor_dot_contrast) 

        all_ft = torch.matmul(_hs, _hs.T)
        all_ft = torch.div(all_ft, self.temperature)  

        mask = 1-torch.eye(self.bsz*2).to(all_ft.device) 
        exp_all = torch.exp(mask * all_ft).sum(1) 
        global_contrast = -torch.log(exp_logits / exp_all).mean()
       
        return global_contrast

    def get_multiscale_patches(self, hs):
        bsz, t, hw, c = hs.shape
        hs = hs.permute(0, 1, 3, 2) 
        hs = hs.reshape(t, bsz, c, 7, 7).flatten(0, 1)  
        hs1 = hs.flatten(2)
        hs2 = self.avgpool_2(hs).flatten(2)
        hs4 = self.avgpool_4(hs).flatten(2)

        hs = torch.cat([hs1, hs2, hs4], dim=2).reshape(t, bsz, c, -1)  
        hs = hs.permute(1, 0, 3, 2)  
        
        return hs

    def get_st_contrast(self, hs, bert_embs):
        global_cts = torch.Tensor([0.]).cuda()
        temp_cts = torch.Tensor([0.]).cuda()
        temp_cycle = torch.Tensor([0.]).cuda()
        spa_cts = torch.Tensor([0.]).cuda()
        spa_cycle = torch.Tensor([0.]).cuda()
        
        bsz = self.bsz * 2 
        hs = hs.permute(1, 0, 2).reshape(bsz, 8, 49, self.d_model)  

        for t in self.temp_set:
            masks = 1 - torch.eye(bsz).cuda()
    
            ths = hs.mean(2) 
            ths = [torch.index_select(ths, 1, p).reshape(bsz, 1, t*self.d_model) for p in self.tuples[t]] 
            ths = torch.hstack(ths) 
            ths =  self.tri_linear(ths) if t == 3 else self.bi_linear(ths)  
            ths = self.temp_linear(ths)
            
            ths = gelu(ths)
            ths = self.norm_temp(ths)
            ths = F.normalize(ths, p=2, dim=2)

            hs = self.spa_linear(hs)
            hs = gelu(hs)
            hs = self.norm_spa(hs)
            
            hs = F.normalize(hs, p=2, dim=3)
            if self.use_spa_mscale:
                shs = self.get_multiscale_patches(hs)  # 48,8,49,2048 -> 48,8,74,2048
            else:
                shs = hs

            cts = torch.Tensor([0.]).cuda()
            cycle_cts = torch.Tensor([0.]).cuda()
            p_cts = torch.Tensor([0.]).cuda()
            p_cycle_cts = torch.Tensor([0.]).cuda()
            
            if self.use_semantic:
                bert_embs = self.emb_fc(bert_embs)
                bert_embs = gelu(bert_embs)
                bert_embs = self.norm_emb(bert_embs)
                bert_embs = F.normalize(bert_embs, p=2, dim=-1)

            for i in range(bsz):  # as query
                t_sims = torch.einsum("ij,jkm->ikm",[ths[i], ths.T])  
                t_sims, y_idxs = torch.max(t_sims, dim=1) 
                _, x_idxs = torch.sort(t_sims, dim=0, descending=True)
                x_idx = x_idxs[:self.topT, :]
                y_idx = torch.gather(y_idxs, 0, x_idx).T 
                x_idx = x_idx.T
 
                x_fts = [torch.index_select(ths[i], dim=0, index=x_idx[j]) for j in range(bsz)]
                y_fts = [torch.index_select(ths[j], dim=0, index=y_idx[j]) for j in range(bsz)]               
                x_fts = torch.stack(x_fts).mean(1)  
                y_fts = torch.stack(y_fts).mean(1)  
                
                xy_sim = torch.einsum("ij,ij->i", [x_fts, y_fts]) * masks[i]  
                pos = i+1 if i%2==0 else i-1
                exp = torch.exp(torch.div(xy_sim, self.temperature))
                cts -= torch.log(exp[pos]/exp.sum())
                
                # temporal cycle
                x_ths = ths[i]  
                y_ths = ths[pos]  
                cycle_sim = torch.einsum("ij,jk->ik",[x_ths,y_ths.T]) 
                cycle_attn = F.softmax(cycle_sim, dim=1)
                merged_y_ths = torch.einsum("ij,jk->ik",[cycle_attn, y_ths])  
                #_, max_ids = torch.max(cycle_attn, dim=1)  
                #merged_y_ths = y_ths[max_ids]  # 56,c
               
                cycle_pos = torch.exp(torch.einsum("ij,ij->i",[merged_y_ths, x_ths])/self.temperature)  
                cycle_all = torch.exp(torch.einsum("ij,jk->ik",[merged_y_ths, x_ths.T])/self.temperature).sum(1) 
                cycle_cts -= torch.log(cycle_pos/cycle_all).mean()
            
                # spatial contrast
                x_pfts, y_pfts = [], []
                for j in range(1):
                    x_tidx = x_idx[:, j] 
                    y_tidx = y_idx[:, j]  
                    x_shs = [torch.index_select(shs[i], 0, self.tuples[t][tid]) for tid in x_tidx]
                    y_shs = [torch.index_select(shs[k], 0, self.tuples[t][tid]) for k, tid in enumerate(y_tidx)]  
                    x_shs = torch.stack(x_shs).flatten(1, 2)  
                    y_shs = torch.stack(y_shs).flatten(1, 2)
                    
                    bsz_p_sims = torch.einsum("ijk,ikm->ijm", [x_shs, y_shs.permute(0, 2, 1)])  # 48, 222, 222 
                    if self.use_semantic:
                        emb_sim = torch.einsum("ij,j->i", [x_shs[0], bert_embs[i]]).unsqueeze(0).repeat(bsz_p_sims.shape[0], 1)  # 48, 222
                        emb_sim = F.softmax(emb_sim, dim=1)
                        bsz_p_sims = emb_sim.unsqueeze(-1)*bsz_p_sims  # 48,222, 222
           
                    p_sims, y_pidxs = torch.max(bsz_p_sims, dim=2) 
                    p_sims, x_pidxs = torch.sort(p_sims, dim=1, descending=True)  

                    x_pidxs = x_pidxs[:, :self.topK]  
                    y_pidxs = torch.gather(y_pidxs, 1, x_pidxs)

                    x_pft = [torch.index_select(x_shs[j], dim=0, index=x_pidxs[j]) for j in range(bsz)]  
                    y_pft = [torch.index_select(y_shs[j], dim=0, index=y_pidxs[j]) for j in range(bsz)]
                    
                    x_pft = torch.stack(x_pft)
                    y_pft = torch.stack(y_pft)
                    x_pfts.append(x_pft)
                    y_pfts.append(y_pft)
                    
                    if self.use_spa_cycle:  # skip or decrease the bsz when OOM 
                        cycle_attn = F.softmax(bsz_p_sims[pos], dim=1)  
                        merged_sy_shs = torch.einsum("ij,jk->ik",[cycle_attn, y_shs[pos]])  
                        #_, max_ids = torch.max(cycle_attn, dim=1)  
                        #merged_sy_shs = y_shs[pos][max_ids]  # 56,c
                        #pdb.set_trace()
                        cycle_pos = torch.exp(torch.einsum("ij,ij->i",[merged_sy_shs, x_shs[pos]])/self.temperature) 
                        cycle_all = torch.exp(torch.einsum("ij,jk->ik",[merged_sy_shs, x_shs[pos].T])/self.temperature).sum(1) 
                        p_cycle_cts -= torch.log(cycle_pos/cycle_all).mean()
                    
                x_pfts = torch.hstack(x_pfts).mean(1) 
                y_pfts = torch.hstack(y_pfts).mean(1)  
                xy_psim = torch.einsum("ij,ij->i", [x_pfts, y_pfts]) * masks[i]  
                
                p_exp = torch.exp(torch.div(xy_psim, self.temperature))
                p_cts -= torch.log(p_exp[pos]/p_exp.sum())
       
            temp_cts += (cts/bsz) 
            temp_cycle += (cycle_cts/bsz)
            spa_cts += (p_cts/bsz)
            spa_cycle += (p_cycle_cts/bsz)

        temp_cts /= len(self.temp_set)
        temp_cycle /= len(self.temp_set)
        spa_cts /= len(self.temp_set)
        spa_cycle /= len(self.temp_set)

        global_cts = self.get_global_contrast(shs)

        return global_cts, temp_cts, temp_cycle, spa_cts, spa_cycle
    
    def get_bert_embeds(self, vids):
        if not self.use_semantic:
            return None
        if self.bert_finetune:
            clsnames = [self.vid2clsname[vid] for vid in vids]
            tokens = self.tokenizer.batch_encode_plus(
                clsnames, add_special_tokens=True,
                padding="longest")["input_ids"]
            bert_embds = self.bert(tokens)
        else:
            emb_ids = [self.vid2embid[vid] for vid in vids]
            bert_embds = self.bert_embeddings[emb_ids]
        return bert_embds

    def forward(self, imgs, vids):
        imgs = imgs.reshape(self.bsz*2*self.seq_len, 3, self.img_size, self.img_size)               
        src = self.encoder(imgs) 
        bert_embs = self.get_bert_embeds(vids)

        # transformer encdoder
        hs, mask, pos_embed = self.st_transformer_encoder(src) 
     
        contrasts = dict()
        contrasts['global'], contrasts["temp"], contrasts["temp_cycle"], \
            contrasts["spatial"], contrasts["spa_cycle"] = self.get_st_contrast(hs, bert_embs)

        return self.action_classifier(hs.mean(0)), contrasts

    def test(self, imgs, vids):
        pivot = self.way * self.shot
        bsz, t, c, h, w = imgs.shape 
        imgs = imgs.view(bsz*t, c, h, w)
        bert_embs = self.get_bert_embeds(vids)

        src = self.encoder(imgs)
        
        hs, mask, pos_embed = self.st_transformer_encoder(src) 
        hs = hs.permute(1, 0, 2).reshape(bsz, 8, 49, self.d_model) 

        sims = torch.zeros([25, self.way]).cuda()        

        temp_sims = torch.zeros([25, self.way]).cuda()
        spa_sims = torch.zeros([25, self.way]).cuda()
        
        for t in self.temp_set:
            num_tuples = len(self.tuples[t])
            ths = hs.mean(2) 
            ths = [torch.index_select(ths, 1, p).reshape(bsz, 1, t*self.d_model) for p in self.tuples[t]] 
            ths = torch.hstack(ths).flatten(0, 1)  
            ths =  self.tri_linear(ths) if t == 3 else self.bi_linear(ths)
            ths = ths.reshape(bsz, num_tuples, self.d_model)  
            ths = F.normalize(ths, p=2, dim=2)
            
            support_ths, query_ths = ths[:pivot], ths[pivot:]  
 
            if self.sim_metric == "cosine":
                t_sims = torch.einsum("ijk,kmn->ijmn", [query_ths, support_ths.T]) 
            else:
                s_ths = support_ths.reshape(-1, self.d_model)  # 5*56, 1152
                q_ths = query_ths.reshape(-1, self.d_model)  # 25*56, 1152
                num_s, num_q = s_ths.shape[0], q_ths.shape[0]
                t_sims = []
                for q in range(num_q):
                    _q_ths = q_ths[q]
                    t_sims.append(-((q_ths[q].repeat(num_s, 1) - s_ths)**2).sum(-1))
                t_sims = torch.stack(t_sims).reshape(25, num_tuples, num_tuples, self.way*self.shot)
            
            # x, y -> query, support
            t_sims, s_idxs = torch.max(t_sims, dim=2) 
            _, q_idxs = torch.sort(t_sims, dim=1, descending=True)
            q_idxs = q_idxs[:, :self.topT, :]  
            s_idxs = torch.gather(s_idxs, 1, q_idxs)  
   
            hs = F.normalize(hs, p=2, dim=3)
            if self.use_spa_mscale:
                shs = self.get_multiscale_patches(hs)  
            else:
                shs = hs
            support_shs, query_shs = shs[:pivot, :, :, :], shs[pivot:, :, :, :] 
            
            if self.use_semantic:
                bert_embs = self.emb_fc(bert_embs)
                bert_embs = gelu(bert_embs)
                bert_embs = self.norm_emb(bert_embs)
                bert_embs = F.normalize(bert_embs, p=2, dim=-1)

            temp_sim, spa_sim = [], []
            for i in range(25): 
                num_s = self.way * self.shot

                q_idx, s_idx = q_idxs[i].T, s_idxs[i].T  
                q_fts = [torch.index_select(query_ths[i], dim=0, index=q_idx[j]) for j in range(pivot)]
                s_fts = [torch.index_select(support_ths[j], dim=0, index=s_idx[j]) for j in range(pivot)]
                
                q_fts = torch.stack(q_fts).mean(1)
                s_fts = torch.stack(s_fts).mean(1) 
                
                if self.sim_metric == "cosine":
                    qs_sim = torch.einsum("ij,ij->i", [q_fts, s_fts]) 
                else:
                    qs_sim = -((q_fts - s_fts)**2).sum(1)
            
                temp_sim.append(qs_sim)
                
                q_pfts, s_pfts = [], []
                for j in range(5):
                    q_tidx = q_idx[:, j]  # 5
                    s_tidx = s_idx[:, j]
                    q_shs = [torch.index_select(query_shs[i], 0, self.tuples[t][tid]) for \
                                tid in q_tidx]  
                    
                    s_shs = [torch.index_select(support_shs[k], 0, self.tuples[t][tid]) for \
                                k, tid in enumerate(s_tidx)]
                    q_shs = torch.stack(q_shs).flatten(1, 2) 
                    s_shs = torch.stack(s_shs).flatten(1, 2) 
                    
                    bsz_p_sims = torch.einsum("ijk,ikm->ijm", [q_shs, s_shs.permute(0, 2, 1)]) 
                    if self.use_semantic:
                        emb_sim = torch.einsum("ij,j->i", [q_shs[j], bert_embs[i]]).unsqueeze(0).repeat(bsz_p_sims.shape[0], 1)  # 48, 222
                        emb_sim = F.softmax(emb_sim, dim=1)
                        bsz_p_sims = emb_sim.unsqueeze(-1)*bsz_p_sims  # 48,222, 222

                    p_sims, s_pidxs = torch.max(bsz_p_sims, dim=2) 
                    p_sims, q_pidxs = torch.sort(p_sims, dim=1, descending=True)  
                    q_pidxs = q_pidxs[:, :self.topK]
                    s_pidxs = torch.gather(s_pidxs, 1, q_pidxs)
                    
                    q_pft = [torch.index_select(q_shs[j], dim=0, index=q_pidxs[j]) for j in range(num_s)]
                    s_pft = [torch.index_select(s_shs[j], dim=0, index=s_pidxs[j]) for j in range(num_s)]

                    q_pft = torch.stack(q_pft)
                    s_pft = torch.stack(s_pft)
                    q_pfts.append(q_pft)
                    s_pfts.append(s_pft)
                
                q_pfts = torch.hstack(q_pfts).mean(1)  
                s_pfts = torch.hstack(s_pfts).mean(1)
                if self.sim_metric == "cosine":
                    qs_psim = torch.einsum("ij,ij->i", [q_pfts, s_pfts])  
                else:
                    qs_psim = -((q_pfts - s_pfts)**2).sum(1)
                
                spa_sim.append(qs_psim)
         
            temp_sims += torch.stack(temp_sim).reshape(self.way*5, self.way, self.shot).mean(2)
            spa_sims += torch.stack(spa_sim).reshape(self.way*5, self.way, self.shot).mean(2)

        temp_sims /= len(self.temp_set)
        spa_sims /= len(self.temp_set)
        sims += self.sigma_temp*temp_sims
        sims += self.sigma_spa*spa_sims

        global_fts = shs.mean(1).mean(1)
        
        support_ft, query_ft = global_fts[:pivot], global_fts[pivot:]
 
        if self.sim_metric == "cosine":
            global_sim = torch.mm(query_ft, support_ft.T)  
            global_sim = global_sim.reshape(25, self.way, self.shot).mean(2)
        else:
            global_sim = Proto(support_ft.unsqueeze(0), query_ft, self.way, self.shot)
        sims += self.sigma_global * global_sim

        query_pred = torch.argmax(sims, dim=1)

        #if self.use_semantic:
        #    embs = action_embs.unsqueeze(1).repeat(1, self.shot, 1).flatten(0, 1)
        #    if self.use_spatial:
        #        semantic_sim = torch.einsum("ijk,km->ijm", [query_ft, embs.T])  # 25,592,2048, 2048,way*shot -> 25,592,way*shot
        #    else:
        #        semantic_sim = torch.einsum("ij,jk->ik", [query_ft, embs.T])  # 25,2048, 2048,way*shot -> 25,way*shot
        #    semantic_sim = semantic_sim.mean(1)
        #    sims += semantic_sim
        
        #sims = torch.pow(sims+1, 2)
        #sims, _ = sims.reshape(-1, self.way, self.shot).max(2)
        #sims = sims.reshape(-1, self.way, self.shot).mean(2)
       
        return query_pred


