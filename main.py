import face_recognition
import cv2 as cv
import numpy as np
from colors import Color
from all_load_image_file import all_load_image_file
from all_face_encodings import all_face_encodings

path = "known-faces"
images, image_names = all_load_image_file(path)
known_face_encodings = all_face_encodings(images)

if not known_face_encodings or not image_names:
    print("❌ No known face data found.")
    exit()

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("📷 Cannot open camera")
    exit()

window_name = "Camera"
cv.namedWindow(window_name, cv.WINDOW_NORMAL)

# Initialize some variables
is_fullscreen = False
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("🖼️ Can't receive frame (stream end?). Exiting ...")
        break

    # Exit the program when the X button is clicked
    if cv.getWindowProperty(window_name, cv.WND_PROP_VISIBLE) < 1:
        break

    # Only process every other frame of video to save time
    if process_this_frame:
        # Resize frame of video to 1/4 size for faster face recognition processing
        small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)

        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)

        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = []
        if face_locations:
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = image_names[best_match_index]
            
            face_names.append(name)

    process_this_frame = not process_this_frame

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv.rectangle(frame, (left, top), (right, bottom), Color.CUSTOM.value, 2)

        # Draw a label with a name
        cv.putText(frame, name, (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 1, Color.CUSTOM.value, 2)

    # Display the resulting frame
    cv.imshow(window_name, frame)

    key = cv.waitKey(1) & 0xFF
    if key in {113, 27}: # Press 'q' or 'Esc' to quit
        break
    elif key == 102: # Press 'f' to toggle fullscreen
        if not is_fullscreen:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
            is_fullscreen = True
        else:
            cv.setWindowProperty(window_name, cv.WND_PROP_FULLSCREEN, cv.WINDOW_NORMAL)
            is_fullscreen = False

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()