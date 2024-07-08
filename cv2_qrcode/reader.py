import csv
import numpy as np
import cv2
from pyzbar.pyzbar import decode
from termcolor import colored

def get_qr_data(input_frame):
    """
    Decode the QR code from the input frame.
    """
    try:
        return decode(input_frame)
    except Exception as e:
        print(f"Error decoding QR code: {e}")
        return []

def draw_polygon(frame, qr_objects):
    """
    Draw a polygon around the QR code and display the decoded text.
    """
    for obj in qr_objects:
        text = obj.data.decode('utf-8')
        points = np.array([obj.polygon], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [points], True, (255, 100, 5), 2)
        cv2.putText(frame, text, (obj.rect.left, obj.rect.top - 10), cv2.FONT_HERSHEY_PLAIN, 2.5, (255, 100, 5), 2)
        print(colored(f"QR Code found: {text}", "green"))
    return frame

def check_device_in_csv(qr_data):
    """
    Check if the QR code data is present in the CSV file.
    """
    with open("devices.csv", mode='r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if qr_data in row:
                return True
    return False

def qrcode_reader():
        """
        Reads QR codes from the video stream captured by the camera.

        This method opens the camera, captures frames, and processes them to detect QR codes.
        It continuously reads frames from the webcam until a QR code containing a device ID is found in a CSV file.
        If a device ID is found, it prints a success message and stops the process.
        If a device ID is not found, it prints a failure message and continues to read frames.
        The processed frames are displayed in a window named "QR Code Scanner".

        Returns:
            None
        """
        cap = cv2.VideoCapture(0)
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                qr_objects = get_qr_data(frame)
                if qr_objects:
                    qr_data = qr_objects[0].data.decode('utf-8')
                    print("QR Code Data:", qr_data)
                    if check_device_in_csv(qr_data):
                        print(colored("Device ID found in CSV file.", "green"))
                        break
                    else:
                        print(colored("Device ID not found in CSV file.", "red"))
                frame = draw_polygon(frame, qr_objects)
                cv2.imshow("QR Code Scanner", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        finally:
            cap.release()
        cv2.destroyAllWindows()


qrcode_reader()





    # def qrcode_reader(self):
    #     """
    #     Reads QR codes from the video stream captured by the camera.

    #     This method opens the camera, captures frames, and processes them to detect QR codes.
    #     It continuously reads frames from the webcam until a QR code containing a device ID is found in a CSV file.
    #     If a device ID is found, it prints a success message, writes the device has been found, and stops the process.
    #     If a device ID is not found, it prints a failure message and continues to read frames.
    #     The processed frames are displayed in a window named "QR Code Scanner".

    #     Returns:
    #         None
    #     """
    #     cap = cv2.VideoCapture(0)
    #     try:
    #         while True:
    #             ret, frame = cap.read()
    #             if not ret:
    #                 print("Failed to grab frame")
    #                 break
    #             qr_objects = get_qr_data(frame)
    #             if qr_objects:
    #                 qr_data = qr_objects[0].data.decode('utf-8')
    #                 print(qr_data)
    #                 if check_device_in_csv(qr_data):
    #                     print(colored("Device ID found in CSV file.", "green"))
    #                     with open('data/devices.csv', mode='a') as file:
    #                         file.write(f"Device with ID '{qr_data}' has been found.\n")
    #                     break
    #                 else:
    #                     print(colored("Device ID not found in CSV file.", "red"))
    #             frame = draw_polygon(frame, qr_objects)
    #             cv2.imshow("QR Code Scanner", frame)
    #             if cv2.waitKey(1) & 0xFF == ord('q'):
    #                 break
    #     finally:
    #         cap.release()
    #         cv2.destroyAllWindows()

