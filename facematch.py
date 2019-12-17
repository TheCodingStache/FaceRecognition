import face_recognition

image_of_me = face_recognition.load_image_file('img/known/Dimitris_Pallas.jpg')
dimitris_face_encoding = face_recognition.face_encodings(image_of_me)[0]

unknown_image = face_recognition.load_image_file('img/unknown/we_will_see.jpg')
unknown_face_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces
results = face_recognition.compare_faces(
    [dimitris_face_encoding], unknown_face_encoding)

if results[0]:
    print('Dimitris is in the picture')
else:
    print('Dimitris is not in the picture')
