from threading import Thread
import numpy as np
import cv2
import time
from matplotlib import pyplot as plt

class WebcamVideoStream:
    def __init__(self):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(1)
        #self.stream.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        #self.stream.set(cv2.CAP_PROP_EXPOSURE, 7.0)
        #cv2.CAP_PROP_EXPOSURE
        (self.grabbed, self.frame) = self.stream.read()

        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True

vs = WebcamVideoStream().start()
#frame = vs.read()

# define range of blue color in HSV
r = 0
g = 0
b = 241
rh = 203
gh = 51
bh = 255
lower_c = np.array([r, g, b])
upper_c = np.array([rh, gh, bh])



def haveSameCoordinates(rect1, rect2):
    if round(r[0][0], 0) == round(r2[0][0], 0) and round(r[0][1], 0) == round(r2[0][1], 0):
        return True
    else:
        return False


def isCorrectRatio(rect):
    if (rect[1][0] > 6 and rect[1][1] > 6):
        correct_ratio = 15.3 / 2.0
        err = 1
        width = rect[1][0]
        height = rect[1][1]
        ratio = round(height / width, 2)
        if ratio < 1:
            ratio = 1 / ratio
        if (ratio > (correct_ratio - err) and ratio < (correct_ratio + err / 2)):
            return True

    return False


def getRegularRatio(ratio):
    r = ratio
    if r < 1:
        r = 1 / r
    return r


KNOWN_DISTANCE = 77
KNOWN_HEIGHT = 183
focalHeight = 51.333336


def distance_to_camera(pixHeight):
    global KNOWN_HEIGHT, focalHeight
    return (KNOWN_HEIGHT * focalHeight) / pixHeight

while(1):
    frame = vs.read()
    # ---- FILTER OUT THINGS WE DON'T WANT ----
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_c, upper_c)
    rgb = cv2.cvtColor(mask, cv2.COLOR_BAYER_BG2RGB)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    # gray = cv2.GaussianBlur(gray, (5, 5), 3)
    edged = cv2.Canny(gray, 35, 135)
    cv2.imshow("contours", edged)

    _, cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    rectborders = [cv2.minAreaRect(c) for c in cnts]
    rounded = []
    pairs = []


    # ---- FILTER OUT REPEAT CONTOURS ----
    for rect in rectborders:
        n = -1
        rnd_rect = (
        (round(rect[0][0], n), round(rect[0][1], n)), (round(rect[1][0], n), round(rect[1][1], n)), round(rect[2], n))
        rounded.append(rnd_rect)
        count = 0
        for r2 in rounded:
            if rnd_rect == r2:
                count += 1
        if count > 1:
            rectborders.remove(rect)

    # ---- GET PAIRS OF SIMILAR CONTOURS THAT MAY BE TARGET ----
    for r in rectborders:
        # sim_* resembles range of difference between rectangles that is deemed "acceptable" for them to be a pair
        sim_ratio = 0.5
        sim_angle = 2
        sim_area = 0.1
        if isCorrectRatio(r):
            print(r)
            ratio_r = round(r[1][1] / r[1][0], 2)
            ratio_r = getRegularRatio(ratio_r)
            angle_r = round(r[2], 1)
            area_r = r[1][1] * r[1][0]
            for r2 in rectborders:
                if r == r2 or haveSameCoordinates(r, r2):
                    break
                elif isCorrectRatio(r2):
                    ratio_r2 = round(r2[1][1] / r2[1][0], 2)
                    ratio_r2 = getRegularRatio(ratio_r2)
                    angle_r2 = round(r2[2], 1)
                    area_r2 = r2[1][1] * r2[1][0]
                    if (ratio_r2 < ratio_r + sim_ratio and ratio_r2 > ratio_r - sim_ratio):
                        if (angle_r2 < angle_r + sim_angle and angle_r2 > angle_r - sim_angle):
                            if (abs(area_r / area_r2 - 1) < sim_area):
                                pairs.append([r, r2])

    print(pairs)

    # ---- DISPLAY VISUALIZATIONS FOR CONTOURS ----
    min_x = 1920
    max_x = 0
    min_y = 1080
    max_y = 0
    for pair in pairs:
        i = 0
        for rect in pair:
            color = (255, 0, 255)

            width = rect[1][0]
            height = rect[1][1]

            x = rect[0][0]
            y = rect[0][1]


            angle = rect[2]

            ratio = round(height / width, 3)
            if ratio != getRegularRatio(ratio):
                print(ratio)
                ratio = round(getRegularRatio(ratio), 3)
                tmp = height
                height = width
                width = tmp


            if x + width/2 < min_x:
                min_x = int(x - width/2)
            if x - width/2 > max_x:
                max_x = int(x + width/2)
            if y + height/2 < min_y:
                min_y = int(y - height/2)
            if y - height/2 > max_y:
                max_y = int(y + height/2)

            spacing = i * 200
            offset = -75
            cv2.putText(frame, str(ratio), (int(x + spacing + offset), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
            cv2.putText(frame, "w: " + str(round(width, 0)), (int(x + spacing + offset), int(y + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "h: " + str(round(height, 0)), (int(x + spacing + offset), int(y + 40)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, "angle: " + str(round(angle, 0)) + "deg", (int(x + spacing + offset), int(y + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if i == 1:
                inches = distance_to_camera(height)
                cv2.putText(frame, "%.2fft" % (inches / 12), (frame.shape[1] - 200, frame.shape[0] - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)

            cv2.circle(frame, (int(round(x, 0)), int(round(y, 0))), 2, (0, 0, 0), 1)
            box = cv2.boxPoints(rect)
            box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

            cv2.drawContours(frame, [box], -1, color, 2)
            i += 1


        track_window = (min_x, min_y, max_x - min_x, max_y - min_y)
        #Tracking stuff
        """
        print(track_window)
        roi = frame[min_y:max_y, min_x:max_x]
        #print(roi)
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv_roi, lower_c, upper_c)
        #mask_roi = mask[min_y:max_y, min_x:max_x]
        roi_hist = cv2.calcHist([hsv_roi], [0], mask_roi, [256], [0, 256])
        cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1)
        cv2.imshow("mask", mask_roi)

        while True:
            # grab the frame from the threaded video stream and resize it
            # to have a maximum width of 400 pixels
            frame = vs.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.meanShift(dst, track_window, term_crit)
            filter = cv2.inRange(hsv, np.array((0, 146, 149)), np.array((102, 178, 213)))
            x, y, w, h = track_window
            cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
            cv2.rectangle(filter, (x, y), (x + w, y + h), 255, 2)
            cv2.putText(frame, 'Tracked', (x - 25, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)

            # frame = imutils.resize(frame, width=400)

            # check to see if the frame should be displayed to our screen
            obj = frame[y:y + h, x:x + w]
            cv2.imshow("Frame", frame)
            cv2.imshow("obj", obj)

            cv2.imshow("filter", filter)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        """
    cv2.imshow("image", frame)  # cv2.resize(image, (960, 540))
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()