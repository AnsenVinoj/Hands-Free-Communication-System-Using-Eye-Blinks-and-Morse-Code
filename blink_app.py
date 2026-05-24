from flask import Flask, render_template, Response, jsonify
import cv2
import time
import os
from blink_detector import EyeBlinkDetector

app = Flask(__name__)

detector = EyeBlinkDetector()

# Support environment variable for camera index
cam_source = os.getenv("CAMERA_SOURCE", "0")
if cam_source.isdigit():
    cam_source = int(cam_source)
cap = cv2.VideoCapture(cam_source)

@app.route('/')
def index():
    return render_template('blink_demo.html')

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame, blink_detected, ear = detector.process_frame(frame)
        
        # Add visual feedback on the frame
        h, w, _ = frame.shape
        cv2.putText(frame, f"Blinks: {detector.total_blinks}", (30, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {ear:.2f}", (30, 90),
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 1)
        
        if detector.counter >= detector.CONSECUTIVE_FRAMES:
            cv2.putText(frame, "BLINK!", (w//2 - 50, h//2),
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stats')
def stats():
    return jsonify({
        "blinks": detector.total_blinks,
        "ear": detector.counter
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
