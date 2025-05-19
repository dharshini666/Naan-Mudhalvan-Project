import cv2
import numpy as np

def is_stop_sign(contour):
    # Approximate the contour shape
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

    # STOP signs are typically octagons (8 sides), but due to angles, we allow 6-9 sides
    if 6 <= len(approx) <= 9:
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize for performance
    resized = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define red color range (tuned for daylight; may need adjusting)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Mask for red areas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small contours
            if is_stop_sign(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized, "STOP Sign Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Stop Sign Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def is_stop_sign(contour):
    # Approximate the contour shape
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

    # STOP signs are typically octagons (8 sides), but due to angles, we allow 6-9 sides
    if 6 <= len(approx) <= 9:
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize for performance
    resized = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define red color range (tuned for daylight; may need adjusting)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Mask for red areas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small contours
            if is_stop_sign(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized, "STOP Sign Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Stop Sign Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def is_stop_sign(contour):
    # Approximate the contour shape
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

    # STOP signs are typically octagons (8 sides), but due to angles, we allow 6-9 sides
    if 6 <= len(approx) <= 9:
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize for performance
    resized = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define red color range (tuned for daylight; may need adjusting)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Mask for red areas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small contours
            if is_stop_sign(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized, "STOP Sign Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Stop Sign Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

def is_stop_sign(contour):
    # Approximate the contour shape
    approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)

    # STOP signs are typically octagons (8 sides), but due to angles, we allow 6-9 sides
    if 6 <= len(approx) <= 9:
        return True
    return False

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Resize for performance
    resized = cv2.resize(frame, (640, 480))
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Define red color range (tuned for daylight; may need adjusting)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    # Mask for red areas
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Find contours in red mask
    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:  # Ignore small contours
            if is_stop_sign(contour):
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(resized, "STOP Sign Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Stop Sign Detection", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()