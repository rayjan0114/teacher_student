from pathlib import Path
import os
import argparse
import sys
import numpy as np
sys.path.append('utils')
from utils import image_files_in_folder, getRGB

def run(pics_path,out_path):
    i = 0
    with open(out_path, 'w+') as f:
        for class_path in os.listdir(pics_path):
            if not os.path.isdir(os.path.join(pics_path, class_path)):
                continue

            for img_path in image_files_in_folder(os.path.join(pics_path, class_path)):
                img = getRGB(img_path)
                # FIXME: 偵測的到臉的才寫路徑
                if img is None:
                    print('not found',i)
                    continue
                f.writelines([img_path,'\n'])
                if i % 100 ==0:
                    print(i)
                i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pics_path', type=str, default=r'C:\\cygwin64\\home\\Ray\\dev\\data\\train\\images\\')
    # parser.add_argument('--out_path', type=str, default=r'data/ma_train.txt')
    parser.add_argument('--pics_path', type=str, default=r'C:\\cygwin64\\home\\Ray\\dev\\data\\faces_glint\\images\\')
    parser.add_argument('--out_path', type=str, default=r'data/glint_train.txt')
    args = parser.parse_args()
    print('starting..')
    run(args.pics_path, args.out_path)
    # gg = getRGB(r'C:\\cygwin64\\home\\Ray\\dev\\data\\train\\images\\003cb849-407b-43a2-aac2-c6ce0f140821\07c050e2-bdb6-11e9-a8e4-0242ac130002.jpg')
    # print(gg)