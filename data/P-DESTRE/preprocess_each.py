import cv2

vidcap = cv2.VideoCapture("videos/13-11-2019-3-2.MP4")

frame_idx = 0
while True:
    success, image = vidcap.read()
    frame_idx += 1
    if not success:
        break
    img_name = f"{frame_idx:06d}.jpg"
    cv2.imwrite(f"images/13-11-2019-3-2/img1/{img_name}", image)     # remove 22-10-2019-1-2 and add 13-11-2019-3-2