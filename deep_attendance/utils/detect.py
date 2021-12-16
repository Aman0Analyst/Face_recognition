from imutils.video import VideoStream
import face_recognition
import argparse
import imutils
import numpy as np
import pickle
import time
import os
import configparser
import cv2
import logging
logger = logging.getLogger()

class DetectFace():
    '''
    A class to keep face detection functions
    '''
    def __init__(self,model = 'hog', resize_factor = None) -> None:
        logger.info("Initializing Detection Service")
        # reading configuration
        config = configparser.RawConfigParser()
        config.read(os.getenv("DEEP_ATTENDANCE_CONFIG"))
        # getting the detection method
        self.model = model or config["face_detection"]["detection_method"]
        # getting encodings of faces recorded in history
        pickle_encodings = config["people_encodings"]['path']
        self.data = pickle.loads(open(pickle_encodings, "rb").read())
        print(self.data)

    def detect_face(self,img) -> str:
        frame = img
        logger.log("Detecting Face")
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(rgb, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = face_recognition.face_locations(rgb,
            model=self.model)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        for encoding in encodings:
		# attempt to match each face in the input image to our known
		# encodings
            matches = face_recognition.compare_faces(self.data["encodings"],
                encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}

                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    # name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1

                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
            
            # update the list of names
            names.append(name)

if __name__ == "__main__":
    DetectFace(model=None)
