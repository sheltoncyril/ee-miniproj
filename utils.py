# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse
import math

import cv2 as cv
import numpy as np
from PIL import Image

fdtct_clsf = cv.CascadeClassifier("./models/haarcascade_frontalface_default.xml")
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16],
]


class PPHumanSeg:
    def __init__(self, modelPath, backendId=0, targetId=0):
        self._modelPath = modelPath
        self._backendId = backendId
        self._targetId = targetId
        print(self._modelPath)
        self._model = cv.dnn.readNet(self._modelPath)
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)
        self._inputNames = ""
        self._outputNames = ["save_infer_model/scale_0.tmp_1"]
        self._currentInputSize = None
        self._inputSize = [192, 192]
        self._mean = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]
        self._std = np.array([0.5, 0.5, 0.5])[np.newaxis, np.newaxis, :]

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model.setPreferableBackend(self._backendId)
        self._model.setPreferableTarget(self._targetId)

    def _preprocess(self, image):
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self._currentInputSize = image.shape
        image = cv.resize(image, (192, 192))
        image = image.astype(np.float32, copy=False) / 255.0
        image -= self._mean
        image /= self._std
        return cv.dnn.blobFromImage(image)

    def infer(self, image):
        # Preprocess
        inputBlob = self._preprocess(image)
        # Forward
        self._model.setInput(inputBlob, self._inputNames)
        outputBlob = self._model.forward()
        # Postprocess
        results = self._postprocess(outputBlob)
        return results

    def _postprocess(self, outputBlob):
        outputBlob = outputBlob[0]
        outputBlob = cv.resize(
            outputBlob.transpose(1, 2, 0), (self._currentInputSize[1], self._currentInputSize[0]), interpolation=cv.INTER_LINEAR
        ).transpose(2, 0, 1)[np.newaxis, ...]
        result = np.argmax(outputBlob, axis=1).astype(np.uint8)
        return result


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=960,
    display_height=540,
    framerate=30,
    flip_method=2,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


def get_args():
    parser = argparse.ArgumentParser(description="PPHumanSeg (https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.2/contrib/PP-HumanSeg)")
    parser.add_argument("--input", "-i", type=str, help="Usage: Set input path to a certain image, omit if using camera.")
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="./models/human_segmentation_pphumanseg_2023mar.onnx",
        help="Usage: Set model path, defaults to human_segmentation_pphumanseg_2023mar.onnx.",
    )
    parser.add_argument(
        "--backend_target",
        "-bt",
        type=int,
        default=0,
        help="""Choose one of the backend-target pair to run this demo:
                            {:d}: (default) OpenCV implementation + CPU,
                            {:d}: CUDA + GPU (CUDA),
                            {:d}: CUDA + GPU (CUDA FP16),
                        """.format(
            *[x for x in range(len(backend_target_pairs))]
        ),
    )
    return parser.parse_args()


def load_color_image(img_path):
    return Image.open(img_path)


def bg_blur(org, img, mask):
    blurred_img = np.asarray(cv.blur(np.asarray(org), (15, 15)))
    rev_mask = np.where((mask == 0) | (mask == 1), mask ^ 1, mask)
    masked = blurred_img * rev_mask
    img_arr = np.where(masked == 0, 1, masked)
    return cv.addWeighted(img_arr, 1, img, 1, 0)


def bg_replace(mask, src_img, bg_img, resize=True, cc=True):
    if resize:
        bg_img = bg_img.resize((src_img.shape[1], src_img.shape[0]))
    if cc:
        bg_img = cv.cvtColor(np.asarray(bg_img), cv.COLOR_BGR2RGB)
    rev_mask = np.where((mask == 0) | (mask == 1), mask ^ 1, mask)
    masked = bg_img * rev_mask
    image_array = np.where(masked == 0, 1, masked)
    return cv.addWeighted(np.asarray(image_array), 1, src_img, 1, 0)


def face_detect(frame):
    gr_img = cv.cvtColor(np.array(frame), cv.COLOR_BGR2GRAY)
    faces = fdtct_clsf.detectMultiScale(gr_img, 1.3, 5)
    return faces


def _distort(img, w1, h1):
    h, w, _ = img.shape
    flex_x = np.zeros((h, w), np.float32)
    flex_y = np.zeros((h, w), np.float32)
    scale_y = 1
    scale_x = 1
    alpha = -1.8
    center_x, center_y = (w1 // 2, h1 // 2)
    radius = h / 5
    for y in range(h):
        delta_y = scale_y * (y - center_y)
        for x in range(w):
            delta_x = scale_x * (x - center_x)
            distance = delta_x * delta_x + delta_y * delta_y

            if distance >= (radius * radius):
                flex_x[y, x] = x
                flex_y[y, x] = y
            else:
                theta = np.arctan2(delta_x, delta_y) + alpha * (radius - math.sqrt(distance)) / radius
                r_sin = math.sqrt(distance) * np.cos(theta)
                r_cos = math.sqrt(distance) * np.sin(theta)
                flex_x[y, x] = r_cos + center_x
                flex_y[y, x] = r_sin + center_y
    return cv.remap(img, flex_x, flex_y, cv.INTER_LINEAR)


def face_distort(frame):
    faces = face_detect(frame)
    if faces.size != 0:
        x, y, w, h = faces[0]
        roi = frame[y : y + h, x : x + w]
        dis_roi = _distort(roi, w, h)
        frame[y : y + h, x : x + w] = dis_roi
    return frame


def face_replace(frame, repl_img):
    gr_repl_img = cv.cvtColor(np.asarray(repl_img), cv.COLOR_BGR2GRAY)
    _, img_thresh = cv.threshold(gr_repl_img, 0, 250, cv.THRESH_BINARY)
    contours, _ = cv.findContours(img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    large_contours = list(filter(lambda c: cv.contourArea(c) > 15000, contours))
    repl_img = cv.cvtColor(np.asarray(repl_img), cv.COLOR_BGR2RGB)
    d = cv.drawContours(repl_img, large_contours, 0, (0, 255, 0), 3)
    diff_img = d - repl_img
    mask1 = np.zeros_like(diff_img)
    mask2 = cv.drawContours(mask1, large_contours, 0, (1, 1, 1), -1)
    extract_cat = repl_img * mask2
    frame = np.asarray(frame)
    f_c = frame.copy()
    faces = face_detect(frame)
    for face in faces:
        x, y, w, h = face
        roi = frame[y : y + h, x : x + w]
        face_mask = cv.resize(mask2, (roi.shape[0], roi.shape[1]))
        extract_cat = cv.resize(extract_cat, (roi.shape[0], roi.shape[1]))
        face_mask = np.where((face_mask == 0) | (face_mask == 1), face_mask ^ 1, face_mask)
        filter_img = roi * face_mask
        replaced_img = cv.addWeighted(filter_img, 1, extract_cat, 1, 0)
        f_c[y : y + h, x : x + w] = replaced_img
    return f_c


class CustomF:
    def __init__(self, gif):
        self.gif = gif
        self.counter = 0
        self.frame_count = int(self.gif.get(cv.CAP_PROP_FRAME_COUNT))

    def next_frame(self):
        self.gif.set(cv.CAP_PROP_POS_FRAMES, self.counter)
        self.counter += 1
        if self.counter >= self.frame_count:
            self.counter = 0
        _, f = self.gif.read()
        return f

    def blend(self, frame, mask):
        g_frame = self.next_frame()
        g_frame = cv.resize(g_frame, (frame.shape[1], frame.shape[0]))
        return bg_replace(mask, frame, g_frame, resize=False, cc=False)
