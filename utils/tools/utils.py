import numpy as np
import cv2
from math import cos, sin
import dlib
import random
import os
import re
import copy

abs_model_root = r'/Users/raychang/Documents/wry_face/models'
predictor_68_point_model = os.path.join(abs_model_root, 'shape_predictor_68_face_landmarks.dat')
pose_predictor_68_point = dlib.shape_predictor(predictor_68_point_model)

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943), (0.0967927109165, 0.575648016728),
    (0.122141515615, 0.691921601066), (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149), (0.531777802068, 1.06080371126),
    (0.641296298053, 1.03981924107), (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421), (0.96111933829, 0.562238253072),
    (0.970579841181, 0.441758925744), (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323), (0.367460241458, 0.203582210627),
    (0.4392945113, 0.233135599851), (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114), (0.8707571886, 0.235293377042),
    (0.51534533827, 0.31863546193), (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668), (0.475501237769, 0.62076344024),
    (0.520712933176, 0.634268222208), (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002), (0.355749724218, 0.303020650651),
    (0.403718978315, 0.33867711083), (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267), (0.73597236153, 0.294721285802),
    (0.782865376271, 0.321305281656), (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073), (0.477677654595, 0.706835892494),
    (0.522732900812, 0.717092275768), (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972), (0.576410514055, 0.835436670169),
    (0.525398405766, 0.841706377792), (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612), (0.523389793327, 0.748924302636),
    (0.571057789237, 0.74332894691), (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)
])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

# : Landmark indices.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]


def findLandmarks(rgbImg, bb):
    """
    Find the landmarks of a face.
    """
    assert rgbImg is not None
    assert bb is not None

    pose_predictor = pose_predictor_68_point
    # pose_predictor = pose_predictor_5_point
    points = pose_predictor(rgbImg, bb)
    return list(map(lambda p: (p.x, p.y), points.parts()))


def align(imgDim, rgbImg, bb=None, landmarks=None, landmarkIndices=INNER_EYES_AND_BOTTOM_LIP, skipMulti=False):
    assert imgDim is not None
    assert rgbImg is not None
    assert landmarkIndices is not None

    if bb is None:
        bb = dlib.rectangle(0, 0, rgbImg.shape[1], rgbImg.shape[0])
        if bb is None:
            return

    if landmarks is None:
        landmarks = findLandmarks(rgbImg, bb)

    npLandmarks = np.float32(landmarks)
    npLandmarkIndices = np.array(landmarkIndices)

    H = cv2.getAffineTransform(npLandmarks[npLandmarkIndices], imgDim * MINMAX_TEMPLATE[npLandmarkIndices])
    thumbnail = cv2.warpAffine(rgbImg, H, (imgDim, imgDim))

    return thumbnail


def point68_list2dict(points):
    return {
        "chin":
            points[0:17],
        "left_eyebrow":
            points[17:22],
        "right_eyebrow":
            points[22:27],
        "nose_bridge":
            points[27:31],
        "nose_tip":
            points[31:36],
        "left_eye":
            points[36:42],
        "right_eye":
            points[42:48],
        "top_lip":
            points[48:55] + [points[64]] + [points[63]] + [points[62]] + [points[61]] + [points[60]],  # noqa: E501
        "bottom_lip":
            points[54:60] + [points[48]] + [points[60]] + [points[67]] + [points[66]] + [points[65]] +
            [points[64]]  # noqa: E501
    }


def draw_point68(img_rgb, point_list, size=2):
    convert_flag = True
    if type(point_list[0]) is tuple:
        convert_flag = False

    # img_bgr = cv.cvtColor(img_rgb, cv.COLOR_RGB2BGR)
    for point in point_list:
        if convert_flag:
            point = tuple(int(p) for p in point)
        img_rgb = cv2.circle(img_rgb, point, size, (0, 255, 0), size)
    # img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    return img_rgb


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=100):

    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def getBGR(path):
    """
        Load the image from disk in BGR format.
        """
    try:
        bgr = cv2.imread(path)
    except Exception:
        bgr = None
    return bgr


def getRGB(path):
    """
    Load the image from disk in RGB format.
    """
    bgr = getBGR(path)
    if bgr is not None:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    else:
        rgb = None
    return rgb


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def img_random(img_rgb):
    img = copy.deepcopy(img_rgb)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # hue, sat, val
    S = img_hsv[:, :, 1].astype(np.float32)  # saturation
    V = img_hsv[:, :, 2].astype(np.float32)  # value
    hsv_s = 0.5703 / 2
    hsv_v = 0.3174 / 2
    a = random.uniform(-1, 1) * hsv_s + 1
    b = random.uniform(-1, 1) * hsv_v + 1

    S *= a
    V *= b

    img_hsv[:, :, 1] = S if a < 1 else S.clip(None, 255)
    img_hsv[:, :, 2] = V if b < 1 else V.clip(None, 255)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
    new_size = (int(b * img.shape[1])), (int(b * img.shape[0]))
    img = cv2.resize(img, new_size, interpolation=cv2.INTER_CUBIC)
    return img


# dist_func = generate_distFunc((492, 273),(496, 273))
def generate_distFunc(point1, point2):
    '''point1 and point2'''
    assert point1[1] != point2[1]
    a, b = np.linalg.lstsq([(point1[1], 1), (point2[1], 1)], [point1[0], point2[0]], rcond=-1)[0]

    def func(point):
        x, y = point
        return np.abs(-x + a * y + b) / np.sqrt(1 + a**2 + 1e-6)

    return func


def bbox_iou(box1, box2, mode='xywh'):
    box1 = np.array(box1)
    box2 = np.array(box2)
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # box2 = box2.T

    # Get the coordinates of bounding boxes
    if mode == 'y1x2y2x1':
        # y1, x2, y2, x1 = box1
        b1_y1, b1_x2, b1_y2, b1_x1 = box1[0], box1[1], box1[2], box1[3]
        b2_y1, b2_x2, b2_y2, b2_x1 = box2[0], box2[1], box2[2], box2[3]
    elif mode == 'x1y1x2y2':
        # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    elif mode == 'xywh':
        # x, y, w, h = box1
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    # min_x2 = np.min((b1_x2, b2_x2))
    # max_x1 = np.max((b1_x1, b2_x1))
    # min_y2 = np.min((b1_y2, b2_y2))
    # max_y1 = np.max((b1_y1, b2_y1))
    # inter_w = (min_x2 - max_x1).clip(0)
    # inter_h = (min_y2 - max_y1).clip(0)
    # inter_area = inter_w * inter_h
    inter_area = (np.min((b1_x2, b2_x2)) - np.max((b1_x1, b2_x1))).clip(0) * \
                 (np.min((b1_y2, b2_y2)) - np.max((b1_y1, b2_y1))).clip(0)

    # Union Area
    union_area = ((b1_x2 - b1_x1) * (b1_y2 - b1_y1) + 1e-16) + \
                 (b2_x2 - b2_x1) * (b2_y2 - b2_y1) - inter_area

    iou = inter_area / union_area  # iou
    return iou


parallel_bbox_iou = np.vectorize(bbox_iou, signature='(n),(n)->()')
