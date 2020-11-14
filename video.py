import cv2
import matplotlib.pyplot as plt
import numpy as np
bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))

# cap = cv2.VideoCapture('with_tripod.mp4')
cap = cv2.VideoCapture("heron.mp4")

success, frame = cap.read()
# Define the codec and create VideoWriter object
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
print(f'Width:{fwidth}, Height:{fheight}')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (fwidth, fheight))

while success:

    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)
    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 600:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow('knn', fg_mask)

    # write the frame
    out.write(frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break
    success, frame = cap.read()


cap.release()
out.release()
cv2.destroyAllWindows()
