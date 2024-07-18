from deepface import DeepFace
import threading
import cv2

# Initialize VideoCapture without specifying backend
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set desired frame width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False
reference_img = cv2.imread('me.jpg')

# Function to perform face verification
def check_face(frame):
    global face_match
    
    try:
        # Perform face verification with DeepFace
        if DeepFace.verify(frame, reference_img.copy())['verified']:
            face_match = True 
        else:
            face_match = False
    except ValueError as e:
        print(f'Error in face verification: {e}')
        face_match = False

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if ret:
        # Process every 30 frames
        if counter % 30 == 0:
            try:
                # Start a new thread for face verification
                threading.Thread(target=check_face, args=(frame.copy(),)).start()
            except ValueError as e:
                print(f'Thread error: {e}')
        
        # Update counter
        counter += 1
        
        # Display face match status on the frame
        if face_match:
            cv2.putText(frame, 'Face Match !', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'No Match !', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow('Video', frame)
        
    # Check for 'q' key press to exit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
