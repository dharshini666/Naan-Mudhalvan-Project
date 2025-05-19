import cv2
import numpy as np
cap = cv2.VideoCapture('nnnn1.mp4')  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(frame, (11, 11), 0)

    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    lower_brown = np.array([5, 50, 50])
    upper_brown = np.array([25, 255, 255])
    mask = cv2.inRange(hsv, lower_brown, upper_brown)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        area = cv2.contourArea(c)
        if area > 1500:  
            x, y, w, h = cv2.boundingRect(c)
            aspect_ratio = w / float(h)

            if 0.5 < aspect_ratio < 3.0: 
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 100), 2)
                cv2.putText(frame, "Possible Animal", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 2)

    cv2.imshow("Color + Contour Animal Detection", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()