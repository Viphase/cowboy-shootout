import cv2
from mp.facade import MediaPipeFacade

cap = cv2.VideoCapture(0)
mp_facade = MediaPipeFacade()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = mp_facade.process_frame(frame)
    cv2.imshow("testik", frame)

mp_facade.close()
cap.release()
cv2.destroyAllWindows()