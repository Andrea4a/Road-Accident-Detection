import cv2
import numpy as np

import torch
from clip import clip
from PIL import Image

def letterbox_to_square(img_bgr: np.ndarray, target: int = 224) -> np.ndarray:
    """
    Resize mantenendo aspect ratio + padding (bande nere) per ottenere target x target.
    Input: img_bgr HxWx3
    Output: target x target x 3
    """
    h, w = img_bgr.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image for letterbox")

    scale = min(target / w, target / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)

    canvas = np.zeros((target, target, 3), dtype=resized.dtype)
    x0 = (target - nw) // 2
    y0 = (target - nh) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def video_crop(video_frame, type):
    l = video_frame.shape[0]
    new_frame = []
    for i in range(l):
        img = cv2.resize(video_frame[i], dsize=(340, 256))
        new_frame.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    #1
    img = np.array(new_frame)
    if type == 0:
        img = img[:, 16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:, :224, :224, :]
    #3
    elif type == 2:
        img = img[:, :224, -224:, :]
    #4
    elif type == 3:
        img = img[:, -224:, :224, :]
    #5
    elif type == 4:
        img = img[:, -224:, -224:, :]
    #6
    elif type == 5:
        img = img[:, 16:240, 58:282, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #7
    elif type == 6:
        img = img[:, :224, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #8
    elif type == 7:
        img = img[:, :224, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #9
    elif type == 8:
        img = img[:, -224:, :224, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #10
    elif type == 9:
        img = img[:, -224:, -224:, :]
        for i in range(img.shape[0]):
            img[i] = cv2.flip(img[i], 1)
    #11 (index 10): full frame (340x256) -> letterbox 224x224
    elif type == 10:
        out = np.zeros((len(video_frame), 224, 224, 3), dtype=img.dtype)
        for i in range(len(video_frame)):
            out[i] = letterbox_to_square(img[i], target=224)
        img = out
    
    return img

def image_crop(image, type):
    img = cv2.resize(image, dsize=(340, 256))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #1
    if type == 0:
        img = img[16:240, 58:282, :]
    #2
    elif type == 1:
        img = img[:224, :224, :]
    #3
    elif type == 2:
        img = img[:224, -224:, :]
    #4
    elif type == 3:
        img = img[-224:, :224, :]
    #5
    elif type == 4:
        img = img[-224:, -224:, :]
    #6
    elif type == 5:
        img = img[16:240, 58:282, :]
        img = cv2.flip(img, 1)
    #7
    elif type == 6:
        img = img[:224, :224, :]
        img = cv2.flip(img, 1)
    #8
    elif type == 7:
        img = img[:224, -224:, :]
        img = cv2.flip(img, 1)
    #9
    elif type == 8:
        img = img[-224:, :224, :]
        img = cv2.flip(img, 1)
    #10
    elif type == 9:
        img = img[-224:, -224:, :]
        img = cv2.flip(img, 1)
    elif type == 10:
        img = letterbox_to_square(img, target=224)
    
    return img

if __name__ == '__main__':
    video = np.zeros([3, 320, 240, 3], dtype=np.uint8)
    corp_video = video_crop(video, 0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/16", device)
    video_features = torch.zeros(0).to(device)
    with torch.no_grad():
        for i in range(video.shape[0]):
            img = Image.fromarray(corp_video[i])
            img = preprocess(img).unsqueeze(0).to(device)
            feature = model.encode_image(img)
            video_features = torch.cat([video_features, feature], dim=0)
    
    video_features = video_features.detach().cpu().numpy()
    np.save('save_path', video_features)
