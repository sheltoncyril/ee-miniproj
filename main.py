import math
import time

import cv2 as cv
import numpy as np

from utils import (
    PPHumanSeg,
    backend_target_pairs,
    bg_blur,
    bg_replace,
    face_detect,
    face_distort,
    face_replace,
    get_args,
    gstreamer_pipeline,
    load_color_image,
)

assert cv.__version__ >= "4.8.0", "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))

args = get_args()
bg_img = load_color_image("./images/bg_repl.jpg")
dog_img = load_color_image("./images/dog.jpg")
star_gif = cv.VideoCapture("./images/starlight.gif")


def custom_filter(frame, count):
    frame_count = int(star_gif.get(cv.CAP_PROP_FRAME_COUNT))
    frames = []
    f = 0
    while True:
        for frame_num in range(frame_count):
            # Read a frame from the video
            ret, frame1 = cap.read()
            frames.append(frame1)
        v_frame = cv.resize(frames[f], (frame.shape[1], frame.shape[0]))
        dstimg = cv.addWeighted(np.array(frame), 1, np.array(v_frame), 1, 0)
    return dstimg


backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]
model = PPHumanSeg(args.model, backend_id, target_id)
deviceId = 1
cap = cv.VideoCapture(deviceId)
frame_count = 0
start_time = time.time()
# cap = cv.VideoCapture(gstreamer_pipeline(flip_method=2), cv.CAP_GSTREAMER)
# w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
# h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
mode = 0
if cap.isOpened():
    window_handle = cv.namedWindow("c_processed_op", cv.WINDOW_AUTOSIZE)
    count = 0
    while cv.getWindowProperty("c_processed_op", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            if mode == 0:
                img = frame
            elif mode == 1 or mode == 2:
                mask = model.infer(frame)
                mask = np.where(np.asarray(mask[0]) != 0, 1, np.asarray(mask[0]))
                mask = np.stack([mask, mask, mask], axis=-1)
                blend_img = frame * mask
                if mode == 1:
                    img = bg_blur(frame, blend_img, mask)
                if mode == 2:
                    img = bg_replace(mask, blend_img, bg_img)
            elif mode == 3:
                img = face_distort(frame)
            elif mode == 4:
                img = face_replace(frame, dog_img)
            count += 1
            # img = custom_filter(frame, count)
            cv.imshow("c_processed_op", img)
        keyCode = cv.waitKey(1)
        if keyCode == ord("q"):
            break
        elif keyCode == ord("0"):
            mode = 0
        elif keyCode == ord("1"):
            mode = 1
        elif keyCode == ord("2"):
            mode = 2
        elif keyCode == ord("3"):
            mode = 3
        elif keyCode == ord("4"):
            mode = 4
        # Increment frame count
        frame_count += 1
        # Calculate FPS every 10 frames
        if frame_count % 10 == 0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            fps = frame_count / elapsed_time
            print(f"FPS: {fps:.2f}")
            frame_count = 0
            start_time = time.time()
    cap.release()
    cv.destroyAllWindows()
else:
    print("Unable to open camera")
