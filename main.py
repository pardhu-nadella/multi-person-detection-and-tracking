import cv2
import numpy as np
import time
import math
import random




'''
USE THE PROPER VIDEO SOURCE
                    0 -> In the case of a laptop, it is a webcam. Otherwise, it is the first detected webcam
"path_to_an_mp4_file" -> A pre recorded video
'''

# videoSource = 0
videoSource = "mainGateTest.mp4"

cap = cv2.VideoCapture(videoSource)


'''
UNCOMMENT LINES 28 AND 29 IF YOU HAVE A GPU AND CUDA.
DEFAULT CONFIGURATION IS TO USE THE CPU.
'''

net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
# net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)






global people
global personId
global colors
global enteredPeople
global exitedPeople
global doorThresh
global doorCoord
people = []
personId = 0
colors = [(255, 0, 0), (0, 255, 0) ,(0, 0, 255)]
enteredPeople = 0
exitedPeople = 0
doorThresh = 50
doorCoord = (670, 400)
appendThresh = 80



class Person:
    def __init__(self, personId, location):
        self.id = personId
        self.curLocation = [x,y]
        self.trajectory = []
        self.state = []
        self.flag = 0
        if personId < len(colors):
            self.color = colors[personId]
        else:
            self.color = colors[personId % len(colors)]
    def addPointToTrajectory(location):
        self.trajectory.append(location)



def plotTrajectories(frame):
    global people
    global personId
    global colors
    global enteredPeople
    global exitedPeople
    global doorThresh
    global doorCoord
    for person in people:
        prev_point = None  # Store the previous point
        if len(person.trajectory) < 5:
            traj = person.trajectory[int(0.8 * len(person.trajectory)) - 1:]
        else:
            traj = person.trajectory[0:5]
        flag = 0
        for i in traj:
            x, y = i[0], i[1]
            color = person.color
            frame = cv2.circle(frame, (x, y), 3, color, cv2.FILLED)
            if calcDist(doorCoord, i)>50:
                flag = 1
            
            if prev_point is not None:
                frame = cv2.line(frame, prev_point, (x, y), color, 1)  # Draw a line to the previous point
            prev_point = (x, y)  # Update the previous point

    return frame



def calcDist(a, b):
    x1 = a[0]
    y1 = a[1]
    x2 = b[0]
    y2 = b[1]
    distance = math.sqrt(((x2-x1)*(x2-x1)) + ((y2-y1)*(y2-y1)))
    return distance

def trackerHandle(curCoords, frame):
    global people
    global personId
    global colors
    global enteredPeople
    global exitedPeople
    global doorThresh
    global doorCoord

    if len(people)>0:
        if len(curCoords) > 0:
            if not(len(people) == len(curCoords)):
                ## EXISTING PERSON EXITED FROM THE FRAME
                if len(people) > len(curCoords):
                    for person in people:
                        while len(people) > len(curCoords):
                            # print("adding")
                            dists = []
                            coords = []
                            for curCoord in curCoords:
                                dists.append(calcDist(curCoord, person.curLocation))
                                coords.append(curCoord)
                            maxIndex = dists.index(max(dists))

                            # ## CHECK THE PREVIOUS 20% OF THE PERSON's TRAJECTORIES AND IDENTIFY IF THE PERSON WALKED TOWARDS OR AWAY
                            # traj = people[maxIndex].trajectory
                            # initialCoord = traj[int(0.8 * len(traj))]
                            # finalCoord = people[maxIndex].trajectory[len[traj] - 1]
                            # initialDistance = calcDist(doorCoord, initialCoord)
                            # finalDistance = calcDist(doorCoord, finalCoord)
                            # if initialDistance<finalDistance:
                            #     enteredPeople += 1
                            people.pop(maxIndex)

                ## NEW PERSON IS ADDED
                if len(people) < len(curCoords):
                    for person in people:
                        while len(people) < len(curCoords):
                            # print("removing")
                            dists = []
                            coords = []
                            for curCoord in curCoords:
                                dists.append(calcDist(curCoord, person.curLocation))
                                coords.append(curCoord)
                            maxIndex = dists.index(max(dists))
                            curCoords.pop(maxIndex)
                            # if len(person.trajectory) == 1:
                            #     if calcDist(doorCoord, person.trajectory[0]) < doorThresh:
                            #         exitedPeople += 1
                            createPerson(coords[maxIndex])


            if len(people) == len(curCoords):
                for person in people:
                    dists = []
                    coords = []
                    for curCoord in curCoords:
                        dists.append(calcDist(curCoord, person.curLocation))
                        coords.append(curCoord)
                    minIndex = dists.index(min(dists))

                    if calcDist(person.curLocation, coords[minIndex]) < appendThresh:
                        person.trajectory.append(coords[minIndex])

                    person.curLocation = coords[minIndex]



                    traj = person.trajectory
                    if len(traj) > 0 and person.flag==0:
                        initialCoord = traj[int(0.8 * len(traj))]
                        finalCoord = person.trajectory[len(traj) - 1]
                        initialDistance = calcDist(doorCoord, initialCoord)
                        finalDistance = calcDist(doorCoord, finalCoord)
                        initialx = initialCoord[0]
                        finalx = finalCoord[0]
                        if (initialx-finalx<50 and finalDistance<80):
                            enteredPeople += 1
                            person.flag = 1
                        elif (initialDistance < 20 and finalDistance > 50):
                            exitedPeople += 1
                            person.flag = 1

                    # print(calcDist(doorCoord, person.curLocation))
                    curCoords.pop(minIndex)
        else:
            people = []

    elif len(people) == 0:
        if len(curCoords)>0:
            for i in curCoords:
                createPerson(i)
        curCoords = []

    frame = plotTrajectories(frame)
    return frame

def createPerson(currentCoord):
    global people
    global personId
    global colors

    person = Person(personId, [currentCoord[0], currentCoord[1]])
    people.append(person)
    personId += 1    







font = cv2.FONT_HERSHEY_PLAIN







processFrame = True
with open("coco.names", "r") as f:
    classes = f.read().strip().split('\n')



## Set the window size (adjust as needed)
cv2.namedWindow("Person Tracking", cv2.WINDOW_NORMAL)


while True:
    startTime = time.time()
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))
    frame = cv2.circle(frame, doorCoord, 5, (255, 0, 0))
    if processFrame == True:
        # Detect objects in the frame
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(net.getUnconnectedOutLayersNames())

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5 and classes[class_id] == 'person':
                    center_x, center_y, w, h = (detection[0:4] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])).astype(int)
                    x, y = int(center_x - w / 2), int(center_y - h / 2)
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)
        curCoords = []
        for i in indices:
            box = boxes[i]
            curCoords.append([box[0]+int(box[2]/2), box[1]+int(box[3]/2)])
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Draw a rectangle around each person
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, 'Person', (x, y - 10), font, 1, (0, 255, 0), 1)

        frame = trackerHandle(curCoords, frame)
    
    


    peopleCount = len(people)

    # Exit the loop when 'q' is pressed
    key = None
    key = cv2.waitKey(1) & 0xFF
    if key  == ord('q'):
        break
    elif key == ord('a'):
        processFrame = not processFrame
    if not processFrame:
        people = []


    endTime = time.time()
    if startTime == endTime:
        fps = 60
    else:
        fps = 1/(endTime-startTime)

    cv2.putText(frame, f"People Count: {peopleCount}", (550, 30), font, 1.5, (0,0,255), 2)
    cv2.putText(frame, f"Entered Count: {enteredPeople}", (550, 60), font, 1.5, (0,0,255), 2)
    cv2.putText(frame, f"Exited Count: {exitedPeople}", (550, 90), font, 1.5, (0,0,255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), font, 1.5, (0,0,255), 2)
    cv2.imshow("Person Tracking", frame)

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
