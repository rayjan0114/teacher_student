import sys
import os
import argparse
from easydict import EasyDict as edict
from pathlib import Path
import cv2
import numpy as np
import mxnet as mx

arcface_root = r'C:/cygwin64/home/Ray/dev'
arcface_path = arcface_root + '/insightface'
# arcface_path = r'C:/cygwin64/home/Ray/dev/insightface'
sys.path.append(arcface_path)
from face_model import FaceModel, get_model


class TeacherLoader:
    def __init__(self,args=None):
        if args is None:
            args = edict()
            args.model = arcface_root + r'/teacher_student/weights/model-r100-ii/model,0'
            args.image_size = '112,112'
            args.ga_model = arcface_root + r'/teacher_student/weights/gamodel-r50/model,0'
            args.gpu = 0
            args.det = 0
            args.flip = 0
            args.threshold = 1.24
        
        oldpath = os.getcwd()
        os.chdir(arcface_path)
        self.face_model = FaceModel(args)
        os.chdir(oldpath)

    def teacher_img_preprocess(self, img):
        aligned_img = self.face_model.get_input(img[...,::-1])  # RGB => BGR => RGB
        return aligned_img


    def arcface_teacher(self):
        def teacher_ebdd(aligned_img):
            ebdd = self.face_model.get_feature(aligned_img)
            return ebdd

        def teacher_ga(aligned_img):
            input_blob = np.expand_dims(aligned_img, axis=0)
            data = mx.nd.array(input_blob)
            db = mx.io.DataBatch(data=(data,))
            self.face_model.ga_model.forward(db, is_train=False)
            ret = self.face_model.ga_model.get_outputs()[0].asnumpy()
            g = ret[:, 0:2].flatten()
            # gender = np.argmax(g)        
            a = ret[:, 2:202].reshape((100, 2))
            # print(np.sum(a,axis=1))  # how to decode a to age ? reshape to (100,2) then sum(argmax(axis=1))
            a = a.flatten()
            # a = np.argmax(a, axis=1)
            # age = int(sum(a))
            return np.hstack((g,a))
        return teacher_ebdd , teacher_ga

if __name__ == "__main__":
    teacher_loader = TeacherLoader()
    teacher_list = teacher_loader.arcface_teacher()
    arcface_root = r'C:/cygwin64/home/Ray/dev'
    img_path =  arcface_root + r'/insightface/samples/Tom_Hanks_54745.png'
    img = cv2.imread(img_path)
    img = teacher_loader.teacher_img_preprocess(img)
    print([teach(img).shape for teach in teacher_list]) 
    
    