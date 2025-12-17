from cv2 import (VideoCapture, putText, imshow,
waitKey, destroyAllWindows, FONT_HERSHEY_SIMPLEX, LINE_AA)
from utils.tracking import MediaPipeFacade
from game_tracking import *
from utils.tracking import Human
from utils.line import Segment, crossRS
def main():
    cap = VideoCapture(0)
    mp_facade = MediaPipeFacade()

    first_player = Human(None, None, (640, 360)) # hands_results pose_results image_shape
    second_player = Human(None, None, (640, 360))# hands_results pose_results image_shape

    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_number % 4 == 0:
            frame, hands_results, pose_results = mp_facade.process_frame(frame, debug=True)
            labels = []


            first_player.left_hand = hands_results[0]
            first_player.right_hand = hands_results[0]
            first_player.pose = pose_results[0].pose_landmarks[0] if len(pose_results[0].pose_landmarks) else None
            first_player.shape = frame.shape

        
            second_player.left_hand = hands_results[1]
            second_player.right_hand = hands_results[1]
            second_player.pose = pose_results[1].pose_landmarks[0] if len(pose_results[1].pose_landmarks) else None
            second_player.img_shape = frame.shape

            if first_player.right_hand is not None and is_pistol(first_player.right_hand, first_player.img_shape):
                text = f"Player 1 Right Hand: GUN"
                if crossRS(first_player.bullet, second_player.collider):
                    print("Игрок 1:Попал")
                else:
                    print("Игрок 1:Непопал")
                    
                color = (0, 255, 0)
            elif first_player.right_hand is not None and is_shield(first_player.left_hand, first_player.img_shape):
                text = f"Player 1 Right Hand: SHIELD"
                color = (0, 255, 0)
            else:
                text = f"Player 1 Right Hand: NOTHING"
                color = (0, 0, 255)

            labels.append((text, color))


            if second_player.right_hand is not None and is_pistol(second_player.right_hand, second_player.img_shape):
                text = f"Player 2 Right Hand: GUN"
                color = (0, 255, 0)
                if crossRS(second_player.bullet, first_player.collider):
                    print("Игрок 2:Попал")
                else:
                    print("Игрок 2:Непопал")

            if second_player.right_hand is not None and is_shield(second_player.left_hand, second_player.img_shape):
                text = f"Player 2 Right Hand: SHIELD"
                color = (0, 255, 0)         

            if second_player.right_hand is None or (not is_pistol(second_player.right_hand, second_player.img_shape) and not is_shield(second_player.left_hand, second_player.img_shape)):
                text = f"Player 2 Right Hand: NOTHING"
                color = (0, 0, 255)


            labels.append((text, color))

            y = 50
            for text, color in labels:
                putText(frame, text, (50, y), FONT_HERSHEY_SIMPLEX, 0.8, color, 2, LINE_AA)
                y += 40
            imshow("kartinka", frame)

        frame_number += 1
        
        key = waitKey(1)
        if key == ord('q'):
            cap.release()
            destroyAllWindows()
            break


if __name__ == "__main__":
    main()
