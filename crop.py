import cv2
import os
output_dir = "C:/Users/dell/PycharmProjects/parking_car_project/clf_data2"
mask_path = "C:/Users/dell/PycharmProjects/parking_car_project/mask2.png"
mask = cv2.imread(mask_path, 0)
analysis = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
(totalLables, label_ids, values, centroid) = analysis
slots = []
for i in range(1, totalLables):
    # area of the component
    area = values[i, cv2.CC_STAT_AREA]
    # extract the coordinate point
    X1 = values[i, cv2.CC_STAT_LEFT]
    Y1 = values[i, cv2.CC_STAT_TOP]
    W = values[i, cv2.CC_STAT_WIDTH]
    H = values[i, cv2.CC_STAT_HEIGHT]
    # coordinate of the bounding box
    pt1 = (X1, Y1)
    pt2 = (X1*W, Y1*H)
    (X, Y) = centroid[i]
    slots.append([X1, Y1, W, H])

video_path = "C:/Users/dell/PycharmProjects/parking_car_project/park2.mp4"
cap = cv2.VideoCapture(video_path)
frame_nmr = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
ret, frame = cap.read()
while ret:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_nmr)
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (mask.shape[1], mask.shape[0]))
        for slot_nmr, slot in enumerate(slots):
            slot = frame[slot[1]:slot[1]+slot[3], slot[0]:slot[0]+slot[2], :]
            cv2.imwrite(os.path.join(output_dir, '{}_{}.jpeg'.format(str(frame_nmr).zfill(8), str(slot_nmr).zfill(8))),
                        slot)
        frame_nmr += 10
