import torch
import numpy as np


class SupConSampler():
    def __init__(self, labels, num_episode, num_supcon):
        self.num_episode = num_episode
        self.num_supcon = num_supcon

        labels = np.array(labels)
        self.indices = []
        for i in range(1, max(labels) + 1):
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices.append(index)

    def __len__(self):
        return self.num_episode
    
    def __iter__(self):
        for i in range(self.num_episode):
            batchs = []
            classes = torch.randperm(len(self.indices))[:self.num_supcon] # bootstrap(class)

            for c in classes:
                l = self.indices[c]
                pos = torch.randperm(len(l))[:2] # bootstrap(shots)
                batchs.append(l[pos])
           
            batchs = torch.stack(batchs).reshape(-1)
            yield batchs


class EpisodeSampler():
    """
    Episode Generation
    """
    def __init__(self, labels, num_episode, way, shot, query):
        self.num_episode = num_episode
        self.way = way
        self.shot = shot
        self.query = query
        self.shots = shot + query

        labels = np.array(labels)
        self.indices = []
        for i in range(1, max(labels) + 1):
            index = np.argwhere(labels == i).reshape(-1)
            index = torch.from_numpy(index)
            self.indices.append(index)

    def __len__(self):
        return self.num_episode
    
    def __iter__(self):
        for i in range(self.num_episode):
            batchs = []
            classes = torch.randperm(len(self.indices))[:self.way] # bootstrap(class)
            for c in classes:
                l = self.indices[c]
                for _ in range(self.shot):
                    pos = torch.randperm(len(l))[0]
                    batchs += [l[pos]]
                
            for c in classes:
                l = self.indices[c]
                for _ in range(self.query):
                    pos = torch.randperm(len(l))[0]
                    batchs += [l[pos]]

            batchs = torch.stack(batchs).reshape(-1)
            
            yield batchs

"""
# way*(shot+query) -> way*shot+way*query
#batchs = batchs.reshape(self.way, self.shots)
#shot = batchs[:, :self.shot].reshape(-1)
#query = batchs[:, self.shot:].reshape(-1)
#batchs = torch.cat([shot, query])
"""