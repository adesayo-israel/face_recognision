import face_recognition as fr
import cv2
import numpy as np
import os

# Function to load images and generate encodings
def load_images_and_encodings(path):
    known_names = []
    known_name_encodings = []

    images = os.listdir(path)
    for image_path in images:
        image = fr.load_image_file(os.path.join(path, image_path))
        encoding = fr.face_encodings(image)[0]

        known_name_encodings.append(encoding)
        known_names.append(os.path.splitext(os.path.basename(image_path))[0].capitalize())

    return known_names, known_name_encodings

# Path to directory with known images
known_images_path = "./train/"

# Load known images and encodings
known_names, known_name_encodings = load_images_and_encodings(known_images_path)

# Path to test image
test_image_path = "./test/test.jpg"
test_image = cv2.imread(test_image_path)

# Face recognition on test image
face_locations = fr.face_locations(test_image)
face_encodings = fr.face_encodings(test_image, face_locations)

# Initialize counters for evaluation
TP = 0
FP = 0
FN = 0
total_comparisons = 0

# Define ground_truth_label with the actual expected label for the test image
ground_truth_label = "expected_label"  # Replace with the actual label

# Loop through each face in the test image
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = fr.compare_faces(known_name_encodings, face_encoding)
    name = ""

    face_distances = fr.face_distance(known_name_encodings, face_encoding)
    best_match = np.argmin(face_distances)

    if matches[best_match]:
        total_comparisons += 1
        if known_names[best_match] == ground_truth_label:
            TP += 1
        else:
            FP += 1
    else:
        FN += 1

 # Draw rectangle and label on the recognized face
    cv2.rectangle(test_image, (left, top), (right, bottom), (0, 0, 255), 2)
    cv2.rectangle(test_image, (left, bottom - 15), (right, bottom), (0, 0, 255), cv2.FILLED)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(test_image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

# Display and save the annotated image
cv2.imshow("Result", test_image)
cv2.imwrite("./output.jpg", test_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Calculate precision, recall, F1 score
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Calculate FAR and FRR if you have impostor attempts and genuine attempts data
FAR = FP / total_comparisons if total_comparisons > 0 else 0
FRR = FN / total_comparisons if total_comparisons > 0 else 0

# Print or store the calculated metrics
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)
print("FAR:", FAR)
print("FRR:", FRR)
