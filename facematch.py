import face_recognition

image_of_brother = face_recognition.load_image_file('img/known/Vassilis_Pallas.jpg')
vassilis_face_encoding = face_recognition.face_locations(image_of_brother)[0]

unknown_image = face_recognition.load_image_file('img/unknown/we_will_see.jpg')
unknown_face_encoding = face_recognition.face_locations(unknown_image)[0]

# Compare faces

results = face_recognition.compare_faces(
    [vassilis_face_encoding], [unknown_face_encoding])

if results[0]:
    print('This is vassilis')
else:
    print('This is NOT vassilis')
