#face recognition working model

import cv2
import face_recognition as fr
cap  = cv2.VideoCapture(1)
image = fr.load_image_file('aheed.jpg')
image1 = fr.load_image_file('aheed1.jpeg')
imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imagergb1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
encodedface = fr.face_encodings(imagergb)[0]
encodedface1 = fr.face_encodings(imagergb1)[0]


known_face = [encodedface, encodedface1]
known_face_name = ['Aheed', "Aheed"]
odd = False
while True:
    ret, frame = cap.read()

    #rgb_frame = frame[:, :, ::-1]
    face_locations = fr.face_locations(frame)
    face_encodings = fr.face_encodings(frame, face_locations)
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = fr.compare_faces(known_face, face_encoding)
        name = "Unknown"
        #face_distances = fr.face_distance(known_face, face_encoding)
       # best_match_index = fr.argmin(face_distances)
        if True in matches:
            best_match_index = matches.index(True)
            name = known_face_name[best_match_index]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
