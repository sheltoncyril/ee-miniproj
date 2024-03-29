import math
import time

import cv2 as cv
import numpy as np

from utils import (
    CustomF,
    PPHumanSeg,
    backend_target_pairs,
    bg_blur,
    bg_replace,
    face_distort,
    face_replace,
    get_args,
    gstreamer_pipeline,
    load_color_image,
)

# check if a newer version of opencv is present
assert cv.__version__ >= "4.8.0", "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# To flip the image, modify the flip_method parameter (0 and 2 are the most common)
print(gstreamer_pipeline(flip_method=2))

# getting args so that the PPhumanSeg model can be run on modes suitable to a laptop and the CUDA accelerated Jetson
args = get_args()

# Loading necessary images
bg_img = load_color_image("./images/bg_repl.jpg")
dog_img = load_color_image("./images/dog.jpg")
star_gif_cap = cv.VideoCapture("./images/starlight.gif")

# parsing args
backend_id = backend_target_pairs[args.backend_target][0]
target_id = backend_target_pairs[args.backend_target][1]
# init model for background replacement
model = PPHumanSeg(args.model, backend_id, target_id)
# device id is used only for laptop development
deviceId = 0

cap = cv.VideoCapture(deviceId)

# uncomment the below line and comment the one above for Jetson
# cap = cv.VideoCapture(gstreamer_pipeline(flip_method=2), cv.CAP_GSTREAMER)

frame_count = 0
start_time = time.time()

# modes define what is done to the image
# setting default mode
mode = 0
# Mode 0: No effects
# Mode 1: Background blur
# Mode 2: Background replacement
# Mode 3: face replacement
# Mode 4: face replacement
# Mode 5: Background blend

if cap.isOpened():
    window_handle = cv.namedWindow("c_processed_op", cv.WINDOW_AUTOSIZE)
    count = 0
    customf = CustomF(star_gif_cap)
    while cv.getWindowProperty("c_processed_op", 0) >= 0:
        ret, frame = cap.read()
        if ret:
            if mode == 0:
                img = frame
            elif mode == 1 or mode == 2 or mode == 5:
                mask = model.infer(frame)
                mask = np.where(np.asarray(mask[0]) != 0, 1, np.asarray(mask[0]))
                mask = np.stack([mask, mask, mask], axis=-1)
                blend_img = frame * mask
                if mode == 1:
                    img = bg_blur(frame, blend_img, mask)
                elif mode == 2:
                    img = bg_replace(mask, blend_img, bg_img)
                elif mode == 5:
                    img = customf.blend(frame, mask)
            elif mode == 3:
                img = face_distort(frame)
            elif mode == 4:
                img = face_replace(frame, dog_img)
            elif mode == 5:
                img = customf.blend(frame)
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
        elif keyCode == ord("5"):
            mode = 5
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
