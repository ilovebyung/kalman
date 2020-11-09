# --------------------------------------------------------------------
# Implements motion prediction using Kalman Filter
# --------------------------------------------------------------------

import cv2
import numpy as np
import sys

# Instantiate OCV kalman filter


class KalmanFilter:

    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)

    def estimate(self, x, y):
        ''' This function estimates the position of the object'''
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        return predicted

# Detect an object


def detect(frame):

    # Set threshold to filter only green color & Filter it
    lowerBound = np.array([130, 30, 0], dtype="uint8")
    upperBound = np.array([255, 255, 90], dtype="uint8")
    greenMask = cv2.inRange(frame, lowerBound, upperBound)

    # Dilate
    kernel = np.ones((5, 5), np.uint8)
    greenMaskDilated = cv2.dilate(greenMask, kernel)
    #cv2.imshow('Thresholded', greenMaskDilated)

    # Find ball blob as it is the biggest green object in the frame
    [nLabels, labels, stats, centroids] = cv2.connectedComponentsWithStats(
        greenMaskDilated, 8, cv2.CV_32S)

    # First biggest contour is image border always, Remove it
    stats = np.delete(stats, (0), axis=0)
    try:
        maxBlobIdx_i, maxBlobIdx_j = np.unravel_index(
            stats.argmax(), stats.shape)

    # This is our ball coords that needs to be tracked
        x = stats[maxBlobIdx_i, 0] + (stats[maxBlobIdx_i, 2]/2)
        y = stats[maxBlobIdx_i, 1] + (stats[maxBlobIdx_i, 3]/2)
        return [x, y]
    except:
        pass

    return [0, 0]


## knn ##
bg_subtractor = cv2.createBackgroundSubtractorKNN(detectShadows=True)
erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))

# Performs required image processing to get object coordinated in the video
cap = cv2.VideoCapture('traffic.mp4')

if(cap.isOpened() == False):
    print('Cannot open input')
    pass

width = int(cap.get(3))
height = int(cap.get(4))

# Create Kalman Filter Object
kalman = KalmanFilter()
predictedCoords = np.zeros((2, 1), np.float32)

while(cap.isOpened()):
    ret, frame = cap.read()

    if(ret == True):
        fg_mask = bg_subtractor.apply(frame)
        _, thresh = cv2.threshold(fg_mask, 244, 255, cv2.THRESH_BINARY)
        cv2.erode(thresh, erode_kernel, thresh, iterations=2)
        cv2.dilate(thresh, dilate_kernel, thresh, iterations=2)

        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) > 1000:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

        [x, y] = detect(frame)
        predictedCoords = kalman.estimate(x, y)

        # Draw Actual coords from segmentation
        cv2.circle(frame, (int(x), int(y)),
                   20, [0, 0, 255], 2, 8)
        cv2.line(frame, (int(x), int(y + 20)),
                 (int(x + 50), int(y + 20)), [100, 100, 255], 2, 8)
        cv2.putText(frame, "Actual", (int(x + 50), int(y + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])

        # Draw Kalman Filter Predicted output
        cv2.circle(frame, (int(predictedCoords[0]), int(predictedCoords[1])), 20, [
            0, 255, 255], 2, 8)
        cv2.line(frame, (int(predictedCoords[0]) + 16, int(predictedCoords[1]) - 15),
                 (int(predictedCoords[0]) + 50, int(predictedCoords[1]) - 30), [100, 10, 255], 2, 8)
        cv2.putText(frame, "Predicted", (int(predictedCoords[0] + 50), int(
            predictedCoords[1] - 30)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [50, 200, 250])
        cv2.imshow('Input', frame)

        if (cv2.waitKey(300) & 0xFF == ord('q')):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()

# Segment the green ball in a given frame


# # Main Function
# def main():

#     processImg = ProcessImage()
#     processImg.DetectObject()


# if __name__ == "__main__":
#     main()

# print('Program Completed!')
