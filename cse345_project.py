
!pip install -q opencv-python-headless mediapipe fer

from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode
import cv2
import numpy as np


def take_photo(filename='photo.jpg', quality=0.8):
    js = Javascript('''
    async function takePhoto(quality) {
        const div = document.createElement('div');
        const capture = document.createElement('button');
        capture.textContent = '📷 Capture';
        div.appendChild(capture);

        const video = document.createElement('video');
        video.style.display = 'block';
        const stream = await navigator.mediaDevices.getUserMedia({video: true});

        document.body.appendChild(div);
        div.appendChild(video);
        video.srcObject = stream;
        await video.play();

        
        google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

        
        await new Promise((resolve) => capture.onclick = resolve);

        const canvas = document.createElement('canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext('2d').drawImage(video, 0, 0);

        stream.getTracks().forEach(track => track.stop());
        div.remove();

        return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
    display(js)
    data = eval_js('takePhoto({})'.format(quality))
    binary = b64decode(data.split(',')[1])
    with open(filename, 'wb') as f:
        f.write(binary)
    return filename

from IPython.display import Image
image_path = take_photo()
Image(image_path)

from google.colab import files
import cv2
import mediapipe as mp
from fer import FER
from matplotlib import pyplot as plt




image = cv2.imread(image_path)
if image is None:
    raise ValueError("Image not loaded. Check path or upload it.")


rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)
emotion_detector = FER(mtcnn=False)  # Use default instead of MTCNN


results = face_mesh.process(rgb)


image_with_mesh = image.copy()
if results.multi_face_landmarks:
    for landmarks in results.multi_face_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            image=image_with_mesh,
            landmark_list=landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=1)
        )


dominant_emotion, score = emotion_detector.top_emotion(rgb)


def feedback(emotion):
    if emotion in ['happy', 'surprise']:
        return "✅ Student appears engaged."
    elif emotion == 'neutral':
        return "😐 Student appears neutral."
    elif emotion in ['sad', 'angry', 'disgust', 'fear']:
        return "⚠️ Student may be disengaged or confused."
    else:
        return "❓ Unable to detect emotion."


if dominant_emotion is not None and score is not None:
    print(f"Detected Emotion: {dominant_emotion} (confidence: {score:.2f})")
    feedback_msg = feedback(dominant_emotion)
else:
    dominant_emotion = "Unknown"
    score = 0.0
    feedback_msg = "❓ Unable to detect emotion."
    print("No emotion detected. Please ensure your face is clearly visible.")

print("Feedback:", feedback_msg)


plt.imshow(cv2.cvtColor(image_with_mesh, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title(f"Emotion: {dominant_emotion} ({score:.2f})")
plt.show()

