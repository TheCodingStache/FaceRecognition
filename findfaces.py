import face_recognition

image = face_recognition.load_image_file('img/unknown/we_will_see.jpg')
face_locations = face_recognition.face_locations(image)

# Array of coordinates for each face
print(f'there are {len(face_locations)} faces in this image')
