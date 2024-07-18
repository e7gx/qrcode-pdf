from deepface import DeepFace
import threading
import cv2
import time
import csv

# Initialize VideoCapture without specifying backend
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0

# Load reference images
reference_images = {
    'Abdullah ': cv2.imread('face_id/me.jpg'),
        'Abdullah ': cv2.imread('face_id/me.jpg'),
    'Abdullah ': cv2.imread('face_id/me.jpg'),

}

# Initialize dictionary to store face match results
face_matches = {name: False for name in reference_images.keys()}

# Log file setup
log_file = 'face_id/detections_log.csv'

# Function to perform face verification
def check_face(face, name):
    global face_matches
    
    try:
        # Perform face verification with DeepFace
        if DeepFace.verify(face, reference_images[name])['verified']:
            face_matches[name] = True 
        else:
            face_matches[name] = False
    except ValueError as e:
        print(f'Error in face verification for {name}: {e}')
        face_matches[name] = False

# Function to log detection
def log_detection(name):
    with open(log_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([name, time.strftime('%Y-%m-%d %H:%M:%S')])

# Create log file with header
with open(log_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Name', 'Time'])

face_detected = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Process every 30 frames
        if counter % 30 == 0:
            # Detect faces in the frame
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Start a new thread for face verification for each detected face
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                for name in reference_images.keys():
                    threading.Thread(target=check_face, args=(face.copy(), name)).start()
        
        # Update counter
        counter += 1
        
        # Check face matches and log detection
        for (x, y, w, h) in faces:
            detected_names = [name for name, match in face_matches.items() if match]
            if detected_names:
                for name in detected_names:
                    cv2.putText(frame, f'{name}!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    log_detection(name)
                    face_detected = True
                    break
            else:
                cv2.putText(frame, 'No Match!', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Video', frame)
        
        # Stop program if a face is detected
        if face_detected:
            break
        
    # Check for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
