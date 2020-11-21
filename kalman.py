import cv2
import numpy as np
import datetime

# --------------------------------------------------------------------
# Kalman Filter predicts position and velocity
# --------------------------------------------------------------------


class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def estimate(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted


# Create Kalman Filter Object
kalman = KalmanFilter()
predictions = np.zeros((2, 1), np.float32)

# set parameters for BackgroundSubtractorKNN
bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
history_length = 20
bg_subtractor.setHistory(history_length)
# remove image sensor noise
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

# capture input
cap = cv2.VideoCapture('70_1.avi')
success, frame = cap.read()

# frame demension
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]

# cut a frame in half
# q1 = int(fshape[0] * 1/4)
# q3 = int(fshape[0] * 3/4)
# roi = frame[q1:q3, :]

fps = int(cap.get(cv2.CAP_PROP_FPS))
print(f'Width:{fwidth}, Height:{fheight}, FPS:{fps}')

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# write the frame
out = cv2.VideoWriter('out.avi', fourcc, 20.0, (fwidth, fheight))

while success:

    # --------------------------------------------------------------------
    # BackgroundSubtractorKNN detects moving objects
    # --------------------------------------------------------------------

    fg_mask = bg_subtractor.apply(frame)
    ret, thresh = cv2.threshold(fg_mask, 100, 255, cv2.THRESH_BINARY)

    # Removing image sensor noise
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    for detected in contours:
        # Set the valid size
        size = cv2.contourArea(detected)

        if (10 < size < 4000):
            x, y, w, h = cv2.boundingRect(detected)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
            print(f'contourArea: {cv2.contourArea(detected)}')

            predictions = kalman.estimate(x, y)

            # Draw Actual coords from segmentation
            cv2.circle(frame, (int(x), int(y)),
                       20, [0, 0, 255], 2, 8)
            cv2.putText(frame, f"Actual x {x}, y {y}", (int(x + 50), int(y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 0, 255])

            # Draw Kalman Filter Predicted output
            cv2.circle(frame, (int(predictions[0]), int(predictions[1])), 20,
                       [0, 255, 255], 2, 8)
            cv2.putText(frame, f"Predicted xv {int(predictions[2])}, yv {int(predictions[3])} ", (int(predictions[0]), int(
                predictions[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255])
            cv2.imshow('Input', frame)

            # Set file name as 'MM/DD/YY HH:MM:SS'
            # out = datetime.datetime.now().strftime("%x %X")

            # write the frame
            out.write(frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break
    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
