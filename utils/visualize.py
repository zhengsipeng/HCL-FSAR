import os
import cv2
import math
import torch
import numpy as np
from pyheatmap.heatmap import HeatMap


def vis_salient_patch(b_pos_sim, b_sim_12, b_imgs, labels, id2clss, iteration, epoch, save_dir):
    """
    pos_sim: b, p_num
    imgs: b, 2, T, C, H, W
    labels: b
    """
    batchsize = labels.shape[0]
    for b in range(batchsize):
        clss = id2clss[labels[b].item()]
        imgs = b_imgs[b]
        sim_11 = b_pos_sim[b].reshape(8, -1)  # p_num = T, 49+16+4
        sim_12 = b_sim_12[b].reshape(8, -1)

        # sim_12 -> img2
        # sim_11 -> img1
        imgs1 = imgs[0]  # 3, 224, 224  
        imgs2 = imgs[1]

        img1 = heatmap(imgs1, sim_11)
        img2 = heatmap(imgs2, sim_12)

        H, W, C = img1.shape
        img = np.zeros([H*2, W, C])
        img[:H, :, :] = img1
        img[H:, :, :] = img2
        
        img_name = '%d_%d_%d_%s.jpg'%(epoch, iteration, b, clss)
        if not os.path.exists('data/vis/%s'%save_dir):
            os.makedirs('data/vis/%s'%save_dir)
        cv2.imwrite('data/vis/%s/%s'%(save_dir, img_name), img)
        assert 1==0

def heatmap(imgs, sims):
    """
    img: T, 3, H, W
    sim: T, 49+16+4
    """
    print(imgs[0])
    print(sims[0])
    assert 1==0
    scale = [1, 2, 4]
    stride = [1, 2, 4]
    T = imgs.shape[0]
    htmaps = []
    for t in range(T):
        img = imgs[t]  # C, H, W
        _, H, W = img.shape
        assert H==W
        sim = sims[t]  # 49+16+4
        weights = []
        total_num = 0

        for i in range(3):  # 3 scales
            num = math.ceil(7/stride[i])**2
            _sim = sim[total_num:total_num+num]
            total_num += num

            sim_len = math.ceil(H/7)*scale[i]
            _weights = torch.zeros([H, W])
            start_h = 0
            for k in range(num):
                start_w = 0
                for j in range(num): 
                    _weights[start_h:min(H, start_h+sim_len), start_w:min(W, start_w+sim_len)]
                    start_w += min(W, int(sim_len/stride[i]))
                start_h += min(H, int(sim_len/stride[i]))
                

            weights.append(_weights)

        weights = torch.stack(weights)  # 3, H, W
        weights, _ = torch.max(weights, dim=0)  # H, W
        weights = weights.cpu().numpy()

        htmap = draw_heatmap(weights)
        htmaps.append(htmap)
    
    # concat all images and heatmap into a big image
    img = concat_htmaps(imgs, htmaps)

    return img


def draw_heatmap(weights):
    """
    weights: H x W
    """
    pmin = np.min(weights)
    pmax = np.max(weights)
    weights = ((weights - pmin) / (pmax - pmin + 0.000001))*255 
    weights = weights.astype(np.uint8)
    weights = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
    return weights


def concat_htmaps(imgs, htmaps):
    """
    imgs: T, 3, H, W
    htmaps: T, H, W
    """
    T, C, H, W = imgs.shape
    imgs = imgs.permute(0, 2, 3, 1)
    imgs = imgs.numpy()
    img = np.zeros([H*2, W*T, C])

    for i in range(T):
        img[:H, W*i:W*(i+1), :] = imgs[i]
        img[H:, W*i:W*(i+1), :] = htmaps[i]

    return img


#def transform_invert(img, transform_train):
#    """
#    
#    """