from pathlib import Path
import os
import shutil
import argparse
import sys
import numpy as np
from utils import image_files_in_folder, getRGB
from multiprocessing import Pool

sys.path.append('..')

def teacher1(x):
    x = np.arange(5).reshape(-1)
    return x*0.1 
def teacher2(x):
    return np.arange(99,95,-1).reshape(-1)


def teacher_data_processer(teacher_loader,pics_dir,skip_thres=None):
    teacher_list = []
    arcface_teachers = teacher_loader.arcface_teacher()
    teacher_list.extend(arcface_teachers)
    pics_dir = str(Path(pics_dir))
    labels_dir = os.path.join(pics_dir, 'labels')
    images_dir = os.path.join(pics_dir, 'images')
    if os.path.isdir(labels_dir):
        # shutil.rmtree(labels_dir)  # FIXME: SKIP MODE
        # os.mkdir(labels_dir)  # FIXME: SKIP MODE
        print('start teaching at {}'.format(labels_dir))

    # skip_num = 0  # TODO
    for class_dir in os.listdir(images_dir):
        if skip_thres is not None:
            skip_num = int(class_dir.split('_')[1])  # FIXME: SKIP MODE
            if skip_num < skip_thres or skip_num > skip_thres + 30000:  # FIXME: SKIP MODE
                continue  # FIXME: SKIP MODE
        if not os.path.isdir(os.path.join(images_dir, class_dir)):
            continue
        print(skip_num)
        labels_dir_uuid = os.path.join(labels_dir, class_dir)
        if not os.path.isdir(labels_dir_uuid):
            os.mkdir(labels_dir_uuid)
        for img_path in image_files_in_folder(os.path.join(images_dir, class_dir)):
            img = getRGB(img_path)
            img = teacher_loader.teacher_img_preprocess(img)
            if img is None:
                continue
            label_to_study = np.hstack([teach(img) for teach in teacher_list])
            
            label_write_dir = os.path.basename(os.path.splitext(img_path)[0] + '.txt')
            label_write_dir = os.path.join(labels_dir_uuid,label_write_dir)

            with open(label_write_dir, 'w+') as f:
                np.savetxt(label_write_dir, (label_to_study,), delimiter=' ')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pics_dir', type=str, default=r'C:/cygwin64/home/Ray/dev/data/faces_glint2',
     help='images and labels should be in the dir')
    parser.add_argument('--SKIP_THRES',type=int, default=None)
    args = parser.parse_args()
    from teacher import TeacherLoader
    # stupid_teacher = [teacher1, teacher2]
    # teacher_data_processer(stupid_teacher, rgs)
    teacher_loader = TeacherLoader()
    teacher_data_processer(teacher_loader, args.pics_dir , args.SKIP_THRES)

    # with Pool(processes=5) as p:
    #     for thres in [6790000,6820000,6850000,6880000,6910000]:
    #         p.apply_async(teacher_data_processer, args=(teacher_loader, args.pics_dir, thres))
    #     p.close()
    #     p.join()
    # python  teacher_data_processor.py --SKIP_THRES 6910000
    # python  teacher_data_processor.py --SKIP_THRES 6880000
    # python  teacher_data_processor.py --SKIP_THRES 6850000
    # python  teacher_data_processor.py --SKIP_THRES 6820000
    # python  teacher_data_processor.py --SKIP_THRES 6790000