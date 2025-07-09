from sanic import Sanic
from sanic.response import html
import asyncio
import pathlib
import cv2
import mediapipe as mp
import time
import numpy as np
from collections import deque
import ml_framework as mf 
from helpers import output_to_gesture

slideshow_root_path = pathlib.Path(__file__).parent.joinpath("slideshow")

# you can find more information about sanic online https://sanicframework.org,
# but you should be good to go with this example code
app = Sanic("slideshow_server")

app.static("/static", slideshow_root_path)


@app.route("/")
async def index(request):
    return html(open(slideshow_root_path.joinpath("slideshow.html"), "r").read())


@app.websocket("/events")
async def emitter(_request, ws):
    print("websocket connection opened")

    # Modell laden
    scaler = mf.StandardScaler.load()
    my_model = mf.Model()
    my_model.add(mf.Dense(160, 128, activation="sigmoid"))  
    my_model.add(mf.Dense(128, 128, activation="sigmoid"))
    my_model.add(mf.Dense(128, 4))                                       
    my_model.load()

    # MediaPipe initialisieren
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose

    # Einstellungen
    show_video = True
    frame_window_size = 10
    cap = cv2.VideoCapture(0)
    pose_data_queue = deque(maxlen=frame_window_size)
    # # ======================== add calls to your model here ======================
    # # uncomment for event emitting demo: the following loop will alternate
    # # emitting events and pausing
    #
    '''
    while True:
        print("emitting 'right'")
        #app.add_signal(event="right")
        await ws.send("right")
        await asyncio.sleep(2)
    
        print("emitting 'rotate'")
        await ws.send("rotate")
        await asyncio.sleep(2)

        print("emitting 'left'")
        await ws.send("left")
        await asyncio.sleep(2)
    '''
    cooldown = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                break

            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                pose_data = [
                    landmarks[13].x, landmarks[13].y, landmarks[13].z, landmarks[13].visibility,  
                    landmarks[15].x, landmarks[15].y, landmarks[15].z, landmarks[15].visibility,  
                    landmarks[14].x, landmarks[14].y, landmarks[14].z, landmarks[14].visibility,  
                    landmarks[16].x, landmarks[16].y, landmarks[16].z, landmarks[16].visibility   
                ]
                looking = landmarks[2].visibility
                pose_data_queue.append(pose_data)

            # Draw landmarks
            if show_video:
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                cv2.imshow('MediaPipe Pose', image)

            if len(pose_data_queue) == frame_window_size and looking > 0.93:
                pose_array = np.array(pose_data_queue).flatten()
                X_scaled = scaler.transform([pose_array])
                gesture = output_to_gesture(my_model.predict(X_scaled))

                if gesture != 'idle'and cooldown == 0:
                    print(f"Predicted Gesture: {gesture}")
                    await ws.send(gesture)   
                    cooldown = 40
                elif cooldown >= 1:
                    cooldown -= 1

            if cv2.waitKey(5) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(host="127.0.0.1", debug=True)
