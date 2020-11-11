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


# set parameters for BackgroundSubtractorKNN
bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
history_length = 20
bg_subtractor.setHistory(history_length)
# remove image sensor noise
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 11))

# capture input
cap = cv2.VideoCapture('heron.mp4')
success, frame = cap.read()

# Define the codec and create VideoWriter object
fshape = frame.shape
fheight = fshape[0]
fwidth = fshape[1]
print(f'Width:{fwidth}, Height:{fheight}')
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# Create Kalman Filter Object
kalman = KalmanFilter()
predictions = np.zeros((2, 1), np.float32)

while success:

    # --------------------------------------------------------------------
    # BackgroundSubtractorKNN detects moving objects
    # --------------------------------------------------------------------

    fg_mask = bg_subtractor.apply(frame)
    _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)

    # Removing image sensor noise
    cv2.erode(thresh, erode_kernel, thresh, iterations=2)
    cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

    contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                      cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        # Set the valid size
        if cv2.contourArea(c) > 2000:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            predictions = kalman.estimate(x, y)

            # Draw Actual coords from segmentation
            cv2.circle(frame, (int(x), int(y)),
                       20, [0, 0, 255], 2, 8)
            cv2.line(frame, (int(x), int(y + 20)),
                     (int(x + 50), int(y + 20)), [100, 100, 255], 2, 8)
            cv2.putText(frame, "Actual", (int(x + 50), int(y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

            # Draw Kalman Filter Predicted output
            cv2.circle(frame, (int(predictions[0]), int(predictions[1])), 20, [
                0, 255, 255], 2, 8)
            cv2.line(frame, (int(predictions[0]) + 16, int(predictions[1]) - 15),
                     (int(predictions[0]) + 50, int(predictions[1]) - 30), [100, 10, 255], 2, 8)
            cv2.putText(frame, f"Predicted xv {int(predictions[2])}, yv {int(predictions[3])} ", (int(predictions[0] + 50), int(
                predictions[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
            cv2.imshow('Input', frame)

            # Set file name as 'MM/DD/YY HH:MM:SS'
            # out = datetime.datetime.now().strftime("%x %X")

            # write the frame
            out = cv2.VideoWriter('out.avi', fourcc, 20.0, (fwidth, fheight))
            out.write(frame)

    k = cv2.waitKey(30)
    if k == 27:  # Escape
        break
    success, frame = cap.read()

cap.release()
cv2.destroyAllWindows()
