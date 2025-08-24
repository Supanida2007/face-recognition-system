import face_recognition
import cv2 as cv

def all_face_encodings(images):
    known_face_encodings = []
    for i, image in enumerate(images):
        face_image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(face_image)
        if encodings:
            known_face_encodings.append(encodings[0])
        else:
            print(f"⚠️ No face found in image index {i}")
    return known_face_encodings