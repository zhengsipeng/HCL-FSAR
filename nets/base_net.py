"""
Meta-learning method sets including MatchNet, RelatioNet, ProtoNet
"""

"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class BaseNet(nn.Module):
    def __init__(self, args):
        super(BaseNet, self).__init__()
        self.dataset = args.dataset
        self.way = args.way
        self.shot = args.shot
        self.query = args.query
        self.seq_len = args.sequence_length
        self.bsz = args.batch_size
        self.img_size = args.frame_size
        self.reverse = args.reverse

        if args.dataset in ["kinetics100", "ssv2_100", "ssv2_100_otam"]:
            self.num_classes = 64+12 
            if self.reverse and args.dataset == "ssv2_100_otam":
                self.num_classes = 91
            if self.reverse and args.dataset == "ssv2_100":
                self.num_classes = 90
        elif args.dataset == "hmdb51":
            self.num_classes = 31+10
        elif args.dataset == "ucf101":
            self.num_classes = 70+10

        self.bn_threshold = args.bn_threshold
 
        self.name = args.backbone
        if self.name == 'resnet18' or self.name == 'resnet34':
            self.dim_in = 512
        else:
            self.dim_in = 2048

        self.sim_metric = args.sim_metric
        self.method = args.method
        self.encoder = self.build_backbone()
        self.action_classifier = nn.Linear(self.dim_in, self.num_classes)

    def build_backbone(self):
        if self.name == "resnet18":
            resnet = models.resnet18(pretrained=True)  
        elif self.name == "resnet34":
            resnet = models.resnet34(pretrained=True)
        elif self.name == "resnet50":
            resnet = models.resnet50(pretrained=True)
            
        if self.method in ['relation', 'hcl']:
            resnet = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        else:
            resnet = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
                resnet.avgpool)

        return resnet

    def distribute_model(self, num_gpus):
        """
        Distribte the backbone over multiple GPUs
        """
        if num_gpus > 1:
            self.encoder.cuda(0)
            self.encoder = torch.nn.DataParallel(self.encoder, device_ids=[i for i in range(num_gpus)])



def LR(support_feats, query_feats, support_labels, query_labels):
    support_feats = support_feats.cpu().detach().numpy()
    query_feats = query_feats.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()
    query_labels = query_labels.cpu().detach().numpy()
    clf = LogisticRegression(penalty='l2', random_state=0, C=1.0, solver='lbfgs',
                                    max_iter=1000, multi_class='multinomial')

    clf.fit(support_feats, support_labels)
    query_pred = clf.predict(query_feats)
    query_pred = torch.from_numpy(query_pred).cuda()
 
    return query_pred


def SVM(support_feats, query_feats, support_labels, query_labels):
    support_feats = support_feats.cpu().detach().numpy()
    query_feats = query_feats.cpu().detach().numpy()
    support_labels = support_labels.cpu().detach().numpy()
    query_labels = query_labels.cpu().detach().numpy()
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto', C=1, kernel='linear',
                                                decision_function_shape='ovr'))
    clf.fit(support_feats, support_labels)
    query_pred = clf.predict(query_feats)
    query_pred = torch.from_numpy(query_pred).cuda()
    return query_pred


def NN(support, support_labels, query):
    support = support.T
    support = support.unsqueeze(0)
    query = query.unsqueeze(2)

    diff = torch.mul(query - support, query - support)
    distance = diff.sum(1)
    min_idx = torch.argmin(distance, dim=1)
    pred = support_labels[min_idx] # [support_labels[idx] for idx in min_idx]
    return pred


def Cosine(support, support_labels, query):
    """Cosine classifier"""
    support_norm = torch.norm(support, p=2, dim=1, keepdim=True)
    support = support / support_norm
    query_norm = torch.norm(query, p=2, dim=1, keepdim=True)
    query = query / query_norm
    
    cosine_distance = torch.mm(query, support.T)
    max_idx = torch.argmax(cosine_distance, dim=1)
    pred = support_labels[max_idx] 

    return pred


def Proto(support, query, way, shot):
    """Protonet classifier"""
    nc = support.shape[-1]

    support = support.reshape(-1, 1, way, shot, nc)
    support = support.mean(dim=3)

    batch_size = support.shape[0]
    query = query.reshape(batch_size, -1, 1, nc)  
    logits = -((query - support)**2).sum(-1)
    logits = logits.squeeze(0)

    return logits 


# ===================
# MatchingNet Module
# ===================
class AttentionalClassify(nn.Module):
    def __init__(self, way, shot):
        self.way = way
        self.shot = shot
        super(AttentionalClassify, self).__init__()
    
    def get_one_hot_label(self, way, shot):
        """
        :param way: way
        :param shot:   shot
        :return a one hot matrix: way*shot, way
        """
        one_hot=torch.zeros((way*shot,way))
        k=0
        for i in range(way*shot):
            one_hot[i][k]=1
            if (i+1) % shot==0:
                k+=1
        return one_hot

    def forward(self, similarities):
        softmax_similarities = F.softmax(similarities, dim=1)
        one_hot_label = self.get_one_hot_label(self.way, self.shot).cuda()
        logits=torch.bmm(softmax_similarities.unsqueeze(0), one_hot_label.unsqueeze(0)).squeeze(0)
        return logits


class BidirectionalLSTM(nn.Module):
    def __init__(self, layer_size, vector_dim, batch_size=1):
        super(BidirectionalLSTM,self).__init__()

        self.hidden_size=layer_size[0]
        self.vector_dim=vector_dim
        self.num_layer=len(layer_size)
        self.lstm=nn.LSTM(input_size=self.vector_dim, num_layers=self.num_layer, hidden_size=self.hidden_size,bidirectional=True)
        self.batch_size=batch_size

    def init_hidden(self):
        return (torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size).cuda(),
                torch.zeros(self.lstm.num_layers * 2, self.batch_size, self.hidden_size).cuda())

    def forward(self, support_feats, query_feats):
        support_feats = support_feats.unsqueeze(1)  # way*shot, 1, C  
        query_feats = query_feats.unsqueeze(0)   # 1, way*query, C 
        
        hidden=self.init_hidden()

        support_output, hidden = self.lstm(support_feats, hidden)  # support_output : way*shot , 1 , C

        query_outputs=[]
        for i in range(query_feats.size(1)):
            query_output, _ = self.lstm(query_feats[:, i, :].unsqueeze(1),hidden)  # query_out：1, way*query, C
            query_outputs.append(query_output)
        query_outputs=torch.stack(query_outputs,dim=0).squeeze()

        return support_output.squeeze(), query_outputs


class DistanceNetwork(nn.Module):
    """
    cos similarity for matching network
    """
    def __init__(self):
        super(DistanceNetwork,self).__init__()

    def forward(self, support_set, query_set):
        """
        :param support_set:  (way*shot), C, 5
        :param query_set:    (way*query), C, 25
        :return:  (way*query), (way* shot) 
        """
        eps=1e-10
        
        sum_support = torch.sum(torch.pow(support_set, 2), 1)  # way*shot, 1
        support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()  # 1/sqrt(x)
        
        similarity = torch.bmm(query_set.unsqueeze(0),support_set.t().unsqueeze(0)).squeeze()  # way*query, way*shot
        #print(similarity.shape)
        similarity = similarity * support_manitude.unsqueeze(0)
        return similarity

# ===================
# RelationNet Module
# ===================
class RelationNetwork(nn.Module):
    def __init__(self, dim_in=4096):
        super(RelationNetwork, self).__init__()
        dim_out1 = int(dim_in/4)
        self.layer1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out1, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out1, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        dim_out2 = int(dim_out1/4)
        self.layer2 = nn.Sequential(
            nn.Conv2d(dim_out1, dim_out2, kernel_size=3, padding=1),
            nn.BatchNorm2d(dim_out2, momentum=1, affine=True),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(dim_out2, dim_out2)
        self.fc2 = nn.Linear(dim_out2, 1)
       
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = torch.sigmoid(self.fc2(out))
        return out