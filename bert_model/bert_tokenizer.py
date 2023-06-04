import os
import sys
import json
import torch
import pickle as pkl
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import pdb


def extract_bert_emb(dbname):
    print("Loading BERT model...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = BertModel.from_pretrained('bert-base-cased').cuda()
    print("Loaded")

    if "ssv2_100" not in dbname:
        with open("action_lists/%s_dirname2cls.txt"%dbname, "r") as f:
           ls = [l.strip() for l in f.readlines()]
        
        dirnames = [l.strip().split(",")[0] for l in ls]
        classes = [l.strip().split(",")[1] for l in ls]
        
        vid2embid = dict()
        for i, dirname in enumerate(dirnames):
            vids = os.listdir("../data/images/%s/%s"%(dbname, dirname))
            for vid in vids:
                vid2embid[vid] = i

        with open("%s_vid2embid.json"%dbname, "w") as f:
            json.dump(vid2embid, f)

        tokens = tokenizer.batch_encode_plus(classes, add_special_tokens=True)["input_ids"]

    else:  # ssv2_100 or ssv2_100_otam
        with open("action_lists/%s_vid2clsname.json"%dbname, "r") as f:
            vid2clsname = json.load(f)
        #with open("action_lists/%s_reverse_vid2clsname.json"%dbname, "r") as f:
        #    rv_vid2clsname = json.load(f)
        vid_lists = []
        name_lists = []
        for k, v in tqdm(vid2clsname.items()):
            vid_lists.append(k)
            name_lists.append(v)
        
        vid2embid = dict(zip(vid_lists, range(len(vid_lists))))

        with open("%s_vid2embid.json"%dbname, "w") as f:
            json.dump(vid2embid, f)

        print("Num of action names: %d"%len(name_lists))

        tokens = tokenizer.batch_encode_plus(name_lists, add_special_tokens=False)["input_ids"]

    embs = []
    num = 0
    model.eval()
    with torch.no_grad():
        for token in tqdm(tokens):
            num += 1
            if num % 1000 == 0:
                torch.cuda.empty_cache()
            embs += model(torch.as_tensor(token).cuda().unsqueeze(0))[-1].cpu()
            
    embs = torch.stack(embs).detach().numpy()
    with open("%s_embs.pkl"%dbname, "wb") as f:
        pkl.dump(embs, f)


if __name__ == "__main__":
    dbname = sys.argv[1]
    extract_bert_emb(dbname)