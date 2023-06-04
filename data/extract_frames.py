import cv2
import os
import sys
import glob
from tqdm import tqdm


def generate_images(dbname):
    for split in ['train', 'test', 'val']:
        with open('splits/%s/%s.list'%(dbname, split), 'r') as f:
            vlist = [l.strip() for l in f.readlines()]

        if dbname not in ["hmdb51", "ucf101"]:
            vid2dict = dict()
            for v in vlist:
                clss, vid = v.split('/')[0], v.split('/')[1]
                vid2dict[vid] = clss
                if not os.path.exists('images/%s/%s'%(dbname, clss)):
                    os.makedirs('images/%s/%s'%(dbname, clss))

        if dbname in ["hmdb51", "ucf101"]:
            video_paths = ["videos/%s/"%dbname+v for v in vlist]
            postfix = "avi"
        else:
            video_paths = glob.glob('videos/%s/%s/*'%(dbname, split))

            if "ssv2_100" in dbname or "ssv2_100_otam" in dbname:
                postfix = "wemb"
            elif dbname in ["hmdb51", "ucf101"]:
                postfix = "avi"
            else:
                postfix = "mp4"
        vnum = 0
        #assert 1==0
        for video_path in tqdm(video_paths):
            vnum += 1
            vid = video_path.split("/")[-1].split('.')[0]
            cap = cv2.VideoCapture(video_path)
            fnum = 0
            if dbname not in ["hmdb51", "ucf101"]:
                clss = vid2dict[vid]
            else:
                clss = video_path.split("/")[-2]

            imdir = 'images/%s/%s/%s'%(dbname, clss, vid)
            if not os.path.exists(imdir):
                os.makedirs(imdir)
            if len(os.listdir(imdir)) > 0:
                continue
            while True:
                success, frame = cap.read()
                if not success:
                    success, frame = cap.read()
                    if not success:
                        break
                impath = 'images/%s/%s/%s/%s.jpg'%(dbname, clss, vid, fnum)
               
                cv2.imwrite(impath, frame)
                fnum += 1


if __name__ == '__main__':
    dbname = sys.argv[1]
    generate_images(dbname)