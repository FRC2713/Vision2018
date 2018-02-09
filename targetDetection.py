from socketserver import ThreadingMixIn
from threading import Thread
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import math
from networktables import NetworkTables
from http.server import BaseHTTPRequestHandler, HTTPServer
import time
from stream import WebcamVideoStream

NetworkTables.initialize(server="roboRIO-2713-frc.local")
vt = NetworkTables.getTable("VisionProcessing")
vt.putNumber("angle", 0)
vt.putNumber("distance", 1)

vs = WebcamVideoStream().start()
final = vs.read()

port = 8087


class CamHandler(BaseHTTPRequestHandler):
    def do_GET(self):

        if self.path.endswith('.mjpg'):
            self.send_response(200)
            self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
            self.end_headers()
            while True:
                try:
                    global final, port
                    img = final
                    # rc, img = capture.read()
                    # if not rc:
                    #    continue
                    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    jpg = Image.fromarray(imgRGB)
                    tmpFile = BytesIO()
                    jpg.save(tmpFile, 'JPEG')
                    self.wfile.write("--jpgboundary".encode())
                    self.send_header('Content-type', 'image/jpeg')
                    self.send_header('Content-length', str(tmpFile.getbuffer().nbytes))
                    self.end_headers()
                    # print(jpg)
                    self.wfile.write(tmpFile.getvalue())

                    # jpg.save(self.wfile, 'JPEG')
                    time.sleep(0.05)
                except KeyboardInterrupt:
                    break
            return
        if self.path.endswith('.html'):
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write('<html><head></head><body>'.encode())
            self.wfile.write(('<img src="http://127.0.0.1:' + str(port) + '/cam.mjpg"/>').encode())
            self.wfile.write('</body></html>'.encode())
            return


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""

def serve():
    server = ThreadedHTTPServer(("", port), CamHandler)
    server.serve_forever()


server_thread = Thread(target=serve, args=())
server_thread.start()

print("mjpeg server started on port " + str(port))

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
        err = 1.5
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


while (1):
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
            (round(rect[0][0], n), round(rect[0][1], n)), (round(rect[1][0], n), round(rect[1][1], n)),
            round(rect[2], n))
        rounded.append(rnd_rect)
        count = 0
        for r2 in rounded:
            if rnd_rect == r2:
                rectborders.remove(rect)
                rounded.remove(rnd_rect)

    # ---- GET PAIRS OF SIMILAR CONTOURS THAT MAY BE TARGET ----
    for r in rectborders:
        # sim_* resembles range of difference between rectangles that is deemed "acceptable" for them to be a pair
        sim_ratio = 0.5
        sim_angle = 2
        sim_area = 0.2
        if isCorrectRatio(r):
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

    # print(pairs)

    """
    This may help in understanding some of the code:
     _____
    |     |
    |     |  * = center, (x, y)
    |     |  _ and | = side of minimum area rect of contour
    |     |  
    |     |  h / w = approx 15.3 / 2.0 (dimension of single target rectangle)
  h |  *  |
    |     |  
    |     |
    |     |
    |     |
    |_____|
       w
    cv2.minAreaRect(contour) = ((x, y), (w, h), angle)
    
    Reason for inverting values at times is due to the fact that width and height may not correlate from one rectangle to another (width and height may be switched)
    
    """

    # ---- DISPLAY VISUALIZATIONS FOR CONTOURS ----
    min_x = 1920
    max_x = 0
    min_y = 1080
    max_y = 0
    distances = []
    for pair in pairs:
        for rect in pair:
            color = (255, 0, 255)

            width = rect[1][0]
            height = rect[1][1]

            x = rect[0][0]
            y = rect[0][1]

            angle = rect[2]

            ratio = round(height / width, 3)
            if ratio != getRegularRatio(ratio):
                ratio = round(getRegularRatio(ratio), 3)
                tmp = height
                height = width
                width = tmp

            if x - width / 2 < min_x:
                min_x = int(x - width / 2)
            if x + width / 2 > max_x:
                max_x = int(x + width / 2)
            if y - height / 2 < min_y:
                min_y = int(y - height / 2)
            if y + height / 2 > max_y:
                max_y = int(y + height / 2)

            black = (0, 0, 0)

            cv2.putText(frame, str(ratio), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(frame, "w: " + str(round(width, 0)), (int(x), int(y + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        black, 1)
            cv2.putText(frame, "h: " + str(round(height, 0)), (int(x), int(y + 40)), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        black, 1)
            cv2.putText(frame, "angle: " + str(round(angle, 0)) + "deg", (int(x), int(y + 60)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, black, 1)

            inches = distance_to_camera(height)
            distances.append(inches)

            cv2.circle(frame, (int(round(x, 0)), int(round(y, 0))), 2, (0, 0, 0), 1)
            box = cv2.boxPoints(rect)
            box = np.array(box).reshape((-1, 1, 2)).astype(np.int32)

            cv2.drawContours(frame, [box], -1, color, 2)

        track_window = (min_x, min_y, max_x - min_x, max_y - min_y)
        # Tracking stuff
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
    # ---- FINDS AVERAGE DISTANCE OF TARGET AND PERSPECTIVE ANGLE ----
    if len(distances) == 2:

        diff = distances[0] - distances[1]  # this gives us the opposite for the triangle
        distance = round((distances[0] + distances[1]) / 24, 1)
        if abs(diff) < 6:  # 6 is the length in inches of the target, this gives u the hypotenuse
            perspective_angle = round(math.degrees(math.asin(diff / 6)), 3)

            vt.putNumber("angle", perspective_angle)
            vt.putNumber("distance", distance)

            cv2.putText(frame, str(perspective_angle),
                        (frame.shape[1] - 200, frame.shape[0]), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
        cv2.putText(frame, "%.2fft" % (distance),
                    (frame.shape[1] - 200, frame.shape[0] - 100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 3)
    final = frame
    cv2.imshow("image", frame)  # cv2.resize(image, (960, 540))
    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

vs.stop()
cv2.destroyAllWindows()
