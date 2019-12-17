import face_recognition
from PIL import Image, ImageDraw

image_of_brother = face_recognition.load_image_file('img/known/Dimitris_Pallas.jpg')
vassilis_face_encoding = face_recognition.face_encodings(image_of_brother)[0]

# Create and array of encodings and names

known_face_encoding = [
    vassilis_face_encoding
]

known_face_name = [
    "Dimitris Pallas"
]

# Load test image
test_image = face_recognition.load_image_file('img/unknown/we_will_see.jpg')

# Find faces test image
face_locations = face_recognition.face_locations(test_image)
face_encoding = face_recognition.face_encodings(test_image, face_locations)

# Convert to PIL format
pil_image = Image.fromarray(test_image)

# Create an ImageDraw instance
draw = ImageDraw.Draw(pil_image)

# Loop through faces in test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encoding):
    matches = face_recognition.compare_faces(known_face_encoding, face_encoding)

    name = "Unknown person"
    # if match
    if True in matches:
        first_match_index = matches.index(True)
        name = known_face_name[first_match_index]
    # Draw box
    draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 0))
    # Draw label
    text_width, text_height = draw.textsize(name)
    draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 0),
                   outline=(0, 0, 0))
    draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))
del draw

pil_image.show()
