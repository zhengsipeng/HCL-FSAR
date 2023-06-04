import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .base_net import BaseNet, Proto
from .base_net import AttentionalClassify, BidirectionalLSTM, DistanceNetwork  # Matching Network
from .base_net import RelationNetwork  # Relation Network


class MetaNets(BaseNet):
    def __init__(self, args, head='mlp', feat_dim=512, num_gpus=4):
        super(MetaNets, self).__init__(args)

        if self.method == 'proto':
            self.classifier = 'Proto'
        elif self.method == 'match':
            # fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
            self.lstm=BidirectionalLSTM(layer_size=[int(self.dim_in/4)], vector_dim=self.dim_in)
            self.dn=DistanceNetwork()
            self.att_classifier = AttentionalClassify(args.way, args.shot)
        elif self.method == 'relation':
            self.relation_network = RelationNetwork(dim_in=self.dim_in*2)

        if head == 'linear':
            self.head = nn.Linear(self.dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(self.dim_in, self.dim_in),
                nn.ReLU(inplace=False),
                nn.Linear(self.dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        self.fc = nn.Sequential(
                nn.Linear(2048, 256),
                nn.ReLU(inplace=False),
                nn.Linear(256, 5)
            )
        self.use_ce = args.use_ce
        
    def forward(self, x, labels=None):
        b, t, c, h, w = x.shape 
        
        x = x.reshape(b*t, c, h, w)
        if self.method != 'relation':
            x = self.encoder(x).reshape(b, t, -1)
        else:
            x = self.encoder(x)
            _, c, h, w = x.shape
            x = x.reshape(b, t, c, h, w)
        x = x.mean(1)  # b, c, h, w

        pivot = self.way * self.shot
        support_feats, query_feats = x[:pivot], x[pivot:] 

        if self.method == 'proto':
            logits = Proto(support_feats, query_feats, self.way, self.shot)

        elif self.method == "cosine":           
            support_feats = F.normalize(support_feats, p=2, dim=1)
            query_feats = F.normalize(query_feats, p=2, dim=1)
            
            logits = torch.mm(query_feats, support_feats.T).reshape(25, self.way, self.shot)
            logits = logits.mean(2)
           
        elif self.method == 'match':
            support_feats, query_feats = self.lstm(support_feats, query_feats)  
            similarities=self.dn(support_feats, query_feats)
            logits = self.att_classifier(similarities)
        elif self.method == 'relation':
            query_feats_ext = query_feats.unsqueeze(1).repeat(1, self.way*self.shot, 1, 1, 1)  # way*query, way*shot
            support_feats_ext = support_feats.unsqueeze(0).repeat(self.way*self.query, 1, 1, 1, 1)  # way*query, way*shot

            relation_pairs = torch.cat((query_feats_ext, support_feats_ext), 2).view(-1, self.dim_in*2, 7, 7)
            relations = self.relation_network(relation_pairs).view(self.way*self.query, self.way*self.shot)
            logits = relations.reshape(-1, 1)

        pred = logits.argmax(1)
 
        return logits, pred
