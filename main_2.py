import cv2 as cv
import mediapipe as mp
import time
import math
import numpy as np
import queue
import utils
import random as rd


# mouse object
class MouseObject:
    def __init__(self, x, y, s):
        self.x = x
        self.y = y
        self.s = s


# mouse state
MOUSE_STATE_NORMAL = 0
MOUSE_STATE_CANCEL=1
MOUSE_STATE_CLICK = 2
MOUSE_STATE_SCROLL_UP=3
MOUSE_STATE_SCROLL_DOWN=4
# constants
CLOSED_EYES_FRAME = 4
FONTS = cv.FONT_HERSHEY_COMPLEX
WINDOW_SIZE_HEIGHT = 1440
WINDOW_SIZE_WIDTH = 2560
# face bounder indices
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
             149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
FACE_NOSE = [1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197, 220, 275, 278, 294, 326, 327, 344, 440]

# lips indices for Landmarks
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 185, 40, 39,
        37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
LOWER_LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS = [185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78]
# Left eyes indices
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

# right eyes indices
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

map_face_mesh = mp.solutions.face_mesh
# camera object
camera = cv.VideoCapture(0)


# landmark detection function
def landmarksDetection(img, results, draw=False):
    img_height, img_width = img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in
                  results.multi_face_landmarks[0].landmark]
    if draw:
        [cv.circle(img, p, 2, (0, 255, 0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks
    return mesh_coord

# Euclidean distance
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes

    cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    # reRatio = rhDistance / rvDistance
    # leRatio = lhDistance / lvDistance
    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance
    ratio = (reRatio + leRatio) / 2

    return reRatio, leRatio, ratio

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # getting the dimension of image
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask
    # cv.imshow('mask', mask)

    # draw eyes image on mask, where white shape is
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys
    # cv.imshow('eyes draw', eyes)
    eyes[mask == 0] = 155

    # getting minium and maximum x and y  for right and left eyes
    # For Right Eye
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item: item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item: item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]
    # returning the cropped eyes
    return cropped_right, cropped_left

# mouse_object = MouseObject(0,0,0)

####  EVENTS

####


def run(mouse_callback):
    start_time = 0
    file = open("demofile.txt", "w")
    file.write("x y\n")
    # variables
    frame_counter = 0
    CEF_COUNTER = 0
    CEF_RIGHT_COUNTER = 0
    TOTAL_BLINKS_RIGHT = 0
    TOTAL_BLINKS = 0
    
    MOUSE_STATE = MOUSE_STATE_NORMAL
    MOUSE_STATE_PREV = MOUSE_STATE_NORMAL
    with map_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        frame_counter = 0

        while True:
            frame_counter += 1  # frame counter
            ret, frame = camera.read()  # getting frame from camera
            if not ret:
                break  # no more frames break
            #  resizing frame

            frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
            frame_height, frame_width = frame.shape[:2]
            rgb_frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            results = face_mesh.process(rgb_frame)
            img_h, img_w, img_c = frame.shape
            face_3d = []
            face_2d = []
            if results.multi_face_landmarks:
                mesh_coords = landmarksDetection(frame, results, False)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in FACE_NOSE], dtype=np.int32)], True, utils.GREEN,
                             1,
                             cv.LINE_AA)
                cv.putText(frame, str(mesh_coords[440]), mesh_coords[440], cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                           cv.LINE_AA)

                reRatio, leRatio, ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                # cv.putText(frame, f'ratio {ratio}', (100, 100), FONTS, 1.0, utils.GREEN, 2)
                utils.colorBackgroundText(frame, f'reRatio : {round(reRatio, 2)}', FONTS, 0.7, (30, 100), 2, utils.PINK,
                                          utils.YELLOW)
                utils.colorBackgroundText(frame, f'leRatio : {round(leRatio, 2)}', FONTS, 0.7, (30, 140), 2, utils.PINK,
                                          utils.YELLOW)

                """
                Nose direction
                """
                for face_landmarks in results.multi_face_landmarks:
                    for idx, lm in enumerate(face_landmarks.landmark):
                        if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                            if idx == 1:
                                nose_2d = (lm.x * img_w, lm.y * img_h)
                                nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 6000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv.Rodrigues(rot_vec)

                # Get angles
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rmat)

                # Get the y rotation degree
                x = angles[0] * 360
                y = angles[1] * 360
                z = angles[2] * 360
                nose_3d_projection, jacobian = cv.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                cv.line(frame, p1, p2, (255, 0, 0), 3)
                # cv.putText(frame, str(p2), p2, cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv.LINE_AA)
                x_ratio = (p2[0] / img_w) * WINDOW_SIZE_WIDTH
                y_ratio = (p2[1] / img_h) * WINDOW_SIZE_HEIGHT
                cv.putText(frame, str(p2) + " : " + str(x_ratio) + '-' + str(y_ratio), p2, cv.FONT_HERSHEY_SIMPLEX, 0.5,
                           (0, 0, 0), 1, cv.LINE_AA)
                file.write(str(x_ratio) + ' ' + str(y_ratio) + "\n")

                if ratio > 3.0:
                    CEF_COUNTER += 1
                    if start_time == 0:
                        start_time = time.time_ns()
                        print(start_time)
                else:
                    if CEF_COUNTER > CLOSED_EYES_FRAME:
                        TOTAL_BLINKS += 1
                        CEF_COUNTER = 0

                
                mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_NORMAL))

                

                if start_time != 0:
                    stop_time = time.time_ns()
                    if stop_time - start_time > 3500000000:
                        if TOTAL_BLINKS == 1:
                            mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_CANCEL ))
                            MOUSE_STATE = MOUSE_STATE_CANCEL
                            MOUSE_STATE_PREV = MOUSE_STATE
                        if TOTAL_BLINKS == 2:
                            mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_CLICK))
                            MOUSE_STATE = MOUSE_STATE_CLICK
                            MOUSE_STATE_PREV = MOUSE_STATE
                        elif TOTAL_BLINKS == 3:
                            mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_SCROLL_UP))
                            MOUSE_STATE = MOUSE_STATE_SCROLL_UP
                            MOUSE_STATE_PREV = MOUSE_STATE
                        elif TOTAL_BLINKS == 4:
                            mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_SCROLL_DOWN))
                            MOUSE_STATE = MOUSE_STATE_SCROLL_DOWN
                            MOUSE_STATE_PREV = MOUSE_STATE
                        else:
                            mouse_callback(MouseObject(x=x_ratio, y=y_ratio, s=MOUSE_STATE_NORMAL))
                            MOUSE_STATE = MOUSE_STATE_NORMAL
                        
                        TOTAL_BLINKS = 0
                        start_time=0

                utils.colorBackgroundText(frame, f'Mouse state previous: {MOUSE_STATE_PREV}', FONTS, 1.0, (40, 310), 2, 8,
                                                  bgColor=(100, 100, 100))
                utils.colorBackgroundText(frame, f'Mouse state: {MOUSE_STATE}', FONTS, 1.0, (40, 270), 2, 8,
                                                  bgColor=(100, 100, 100))


                # mouse_object = MouseObject(int(x_ratio), int(y_ratio), 0)
                # data_queue.put(MouseObject(int(x_ratio), int(y_ratio), s=SEND_TOTAL_BLINKS))

                ###
                
                ###

                utils.colorBackgroundText(frame, f'Blink: {TOTAL_BLINKS}', FONTS, 1.0, (40, 220), 2, 8,
                                              bgColor=(100, 100, 100))


                # if reRatio > 3.0:
                #     CEF_RIGHT_COUNTER += 1
                # else:
                #     if CEF_RIGHT_COUNTER > CLOSED_EYES_FRAME:
                #         TOTAL_BLINKS_RIGHT += 1
                #         CEF_RIGHT_COUNTER = 0

                # cv.putText(frame, f'Total Blinks: {TOTAL_BLINKS}', (100, 150), FONTS, 0.6, utils.GREEN, 2)

                cv.polylines(frame, [np.array([mesh_coords[p] for p in LEFT_EYE], dtype=np.int32)], True, utils.GREEN,
                             1,
                             cv.LINE_AA)
                cv.polylines(frame, [np.array([mesh_coords[p] for p in RIGHT_EYE], dtype=np.int32)], True, utils.GREEN,
                             1,
                             cv.LINE_AA)

                # Blink Detector Counter Completed
                right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                left_coords = [mesh_coords[p] for p in LEFT_EYE]
                crop_right, crop_left = eyesExtractor(frame, right_coords, left_coords)
                # cv.imshow('right', crop_right)
                # cv.imshow('left', crop_left)
                # utils.colorBackgroundText(frame, f'R: {TOTAL_BLINKS_RIGHT}', FONTS, 1.0, (40, 220), 2, 8, bgColor=(100, 100, 100))
                # utils.colorBackgroundText(frame, f'L: {TOTAL_BLINKS_LEFT}', FONTS, 1.0, (40, 320), 2,  8,  bgColor=(100, 100, 100))

            # calculating  frame per seconds FPS
            end_time = time.time() - start_time
            fps = frame_counter / end_time

            frame = utils.textWithBackground(frame, f'FPS: {round(fps, 1)}', FONTS, 1.0, (30, 50), bgOpacity=0.9,
                                             textThickness=2)
            # writing image for thumbnail drawing shape
            # cv.imwrite(f'img/frame_{frame_counter}.png', frame)
            cv.imshow('frame', frame)
            key = cv.waitKey(2)
            if key == ord('q') or key == ord('Q'):
                break
        cv.destroyAllWindows()
        camera.release()

def test(mouse_callback):
    while True:
        idx = rd.randint(0, 3)

        if idx == 0:
            mouse_callback(MouseObject(rd.randint(300,600),rd.randint(300,600),MOUSE_STATE_NORMAL))
        elif idx == 1:
            mouse_callback(MouseObject(rd.randint(300,600),rd.randint(300,600),MOUSE_STATE_CLICK))
        elif idx == 2:
            mouse_callback(MouseObject(rd.randint(300,600),rd.randint(300,600),MOUSE_STATE_SCROLL))
        else:
            pass

#run(data_queue=queue.Queue())

#test()