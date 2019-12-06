import os
import re
import cv2

def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]

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

# def plot_images(imgs, targets, paths=None, fname='images.jpg'):
#     # Plots training images overlaid with targets
#     imgs = imgs.cpu().numpy()
#     targets = targets.cpu().numpy()
#     # targets = targets[targets[:, 1] == 21]  # plot only one class

#     fig = plt.figure(figsize=(10, 10))
#     bs, _, h, w = imgs.shape  # batch size, _, height, width
#     bs = min(bs, 16)  # limit plot to 16 images
#     ns = np.ceil(bs ** 0.5)  # number of subplots

#     for i in range(bs):
#         boxes = xywh2xyxy(targets[targets[:, 0] == i, 2:6]).T
#         boxes[[0, 2]] *= w
#         boxes[[1, 3]] *= h
#         plt.subplot(ns, ns, i + 1).imshow(imgs[i].transpose(1, 2, 0))
#         plt.plot(boxes[[0, 2, 2, 0, 0]], boxes[[1, 1, 3, 3, 1]], '.-')
#         plt.axis('off')
#         if paths is not None:
#             s = Path(paths[i]).name
#             plt.title(s[:min(len(s), 40)], fontdict={'size': 8})  # limit to 40 characters
#     fig.tight_layout()
#     fig.savefig(fname, dpi=200)
#     plt.close()