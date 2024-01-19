# Import required modules
import PySimpleGUI as sg
import cv2 as cv
import time


def getFaceBox(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, bboxes


# set variables
faceProto = r"opencv_face_detector.pbtxt"
faceModel = r"opencv_face_detector_uint8.pb"
ageProto = r"age_deploy.prototxt"
ageModel = r"age_net.caffemodel"
genderProto = r"gender_deploy.prototxt"
genderModel = r"gender_net.caffemodel"
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# get the pic
fname = sg.popup_get_file("choose picture", multiple_files=False, file_types=(("jpeg files", "*.jpg*"),))
if not fname:
    sg.popup("Cancel", "no filename supplied")
    raise SystemExit("Cancelling ")

# Load network and set backend as cpu
ageNet = cv.dnn.readNet(ageModel, ageProto)
genderNet = cv.dnn.readNet(genderModel, genderProto)
faceNet = cv.dnn.readNet(faceModel, faceProto)
ageNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
genderNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)
faceNet.setPreferableBackend(cv.dnn.DNN_TARGET_CPU)

# use the image
cap = cv.VideoCapture(fname if fname else 0)
padding = 20
while cv.waitKey(1) < 0:
    # Read frame
    t = time.time()
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameFace, bboxes = getFaceBox(faceNet, frame)
    if not bboxes:
        print("No face Detected, Checking next frame")
        continue

    for bbox in bboxes:
        # print(bbox)
        face = frame[max(0, bbox[1] - padding):min(bbox[3] + padding, frame.shape[0] - 1),
               max(0, bbox[0] - padding):min(bbox[2] + padding, frame.shape[1] - 1)]

        blob = cv.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        # child or not
        result = age[1:3]
        try:
            result = int(result)
        except:
            result = age[1]
            result = int(result)
        if result > 18:
            sg.popup_notify("The person is Adult", display_duration_in_ms=1000)
        else:
            sg.popup_notify("The person is Child", display_duration_in_ms=1000)

        label = "{},{}".format(gender, age)
        cv.putText(frameFace, label, (bbox[0], bbox[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2,
                   cv.LINE_AA)
        cv.imshow("Age Gender Demo", frameFace)
    print("time : {:.3f}".format(time.time() - t))
