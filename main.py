import streamlit as st
import matplotlib.pyplot as plt
import time
import cv2
import numpy as np
from utile import get_parking_spots_bboxes, empty_or_not


# creating an empty place-holder for displaying video frames
st.title("MALL PARKING LOT AREA CONTROLLER")
st.header("please hit the button if you are looking for an empty parking spot")

import os
os.chdir("C:/Users/dell/PycharmProjects/parking_car_project")

video_placeholder = st.empty()
text_empty = st.empty()

def process_video():

    def calc_diff(im1, im2):
        return np.abs(np.mean(im1) - np.mean(im2))

    mask = 'C:/Users/dell/PycharmProjects/parking_car_project/mask_1920_1080.png'
    video_path = 'C:/Users/dell/PycharmProjects/parking_car_project/parking_1920_1080_loop.mp4'

    mask = cv2.imread(mask, 0)

    cap = cv2.VideoCapture(video_path)

    connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

    spots = get_parking_spots_bboxes(connected_components)

    spots_status = [None for j in spots]
    diffs = [None for j in spots]

    previous_frame = None

    frame_nmr = 0
    ret = True
    step = 30
    while ret:
        ret, frame = cap.read()

        if frame_nmr % step == 0 and previous_frame is not None:
            for spot_index, spot in enumerate(spots):
                x1, y1, w, h = spot

                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])

            print([diffs[j] for j in np.argsort(diffs)][::-1])

        if frame_nmr % step == 0:
            if previous_frame is None:
                arr_ = range(len(spots))
            else:
                arr_ = [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
            for spot_index in arr_:
                spot = spots[spot_index]
                x1, y1, w, h = spot

                spot_crop = frame[y1:y1 + h, x1:x1 + w, :]

                spot_status = empty_or_not(spot_crop)

                spots_status[spot_index] = spot_status

        if frame_nmr % step == 0:
            previous_frame = frame.copy()

        for spot_index, spot in enumerate(spots):
            spot_status = spots_status[spot_index]
            x1, y1, w, h = spots[spot_index]

            if spot_status:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            else:
                frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

        cv2.rectangle(frame, (80, 20), (550, 80), (0, 0, 0), -1)
        cv2.putText(frame, 'Available spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))), (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        text_empty.text(f"Available spots: {sum(spots_status)} / {len(spots_status)}")
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.imshow('frame', frame)
        video_placeholder.image(frame, channels='BGR')
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        frame_nmr += 1
    return spots_status
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    if st.button("Looking for an empty spot"):
        with st.spinner('Wait for it...'):
            time.sleep(5)
        st.success('Done!')
        process_video()

