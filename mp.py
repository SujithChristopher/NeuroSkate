import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize VideoCapture.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image_rgb.flags.writeable = False

    # Process the image with MediaPipe Pose.
    results = pose.process(image_rgb)
import cv2
import mediapipe as mp

# Initialize MediaPipe Pose.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize VideoCapture.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the BGR image to RGB.
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # To improve performance, optionally mark the image as not writeable to pass by reference.
    image_rgb.flags.writeable = False

    # Process the image with MediaPipe Pose.
    results = pose.process(image_rgb)

    # Draw the pose landmarks on the image.
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image with pose landmarks.
    cv2.imshow('MediaPipe Pose', image)

    # Break the loop if 'q' key is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

    # Draw the pose landmarks on the image.
    if results.pose_landmarks:
        mp_drawing = mp.solutions.drawing_utils 
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Display the image with pose landmarks.
    cv2.imshow('MediaPipe Pose', image)

    # Break the loop if 'q' key is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        b
