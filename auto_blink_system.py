from blink_detector import EyeBlinkDetector
import camera_morse
import next_word_predictor
import gemin_predictor
import cv2
import time

class AutoMorseCam(camera_morse.MorseCam):
    def __init__(self, predictor):
        super().__init__(predictor)
        # We override the YOLO model with our new MediaPipe detector
        self.blink_detector = EyeBlinkDetector()
        print("🚀 AutoMorseCam Initialized with MediaPipe Detector")

    def generate_frames(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # Use MediaPipe instead of YOLO
            frame, blink_active, ear = self.blink_detector.process_frame(frame)
            current_time = time.time()
            
            # MediaPipe tells us if an eye is "detected" based on face landmarks
            self.eye_detected = ear > 0
            
            # Blink Logic (Integrated with Morse logic)
            if ear < self.blink_detector.EAR_THRESHOLD and self.eye_detected:
                if self.blink_start is None:
                    self.blink_start = current_time
                else:
                    duration = current_time - self.blink_start
                    if duration >= self.LONG_HOLD_REFRESH:
                        self.update_suggestions()
                        self.blink_start = current_time
            else:
                if self.blink_start:
                    duration = current_time - self.blink_start
                    self.current_symbol += "." if duration < self.DOT_THRESHOLD else "-"
                    self.blink_start = None
                    self.last_blink_end = current_time

            # The rest of the Morse logic is inherited or copied
            pause = current_time - self.last_blink_end
            if self.current_symbol and pause > self.WORD_PAUSE:
                self.morse_code += self.current_symbol + " "
                if self.current_symbol == ".....":
                    if self.decoded_text and not self.decoded_text.endswith(" "):
                        self.decoded_text += " "
                elif self.current_symbol in ["....-", "-....", "--..."]:
                    idx = ["....-", "-....", "--..."].index(self.current_symbol)
                    if idx < len(self.suggestions):
                        word = self.suggestions[idx]
                        if self.decoded_text and not self.decoded_text.endswith(" ") and " " in self.decoded_text.rstrip():
                            prefix = self.decoded_text.rstrip().rsplit(" ", 1)[0]
                            self.decoded_text = prefix + " " + word
                        else:
                            self.decoded_text = word
                elif self.current_symbol == "......": # Backspace
                    self.decoded_text = self.decoded_text[:-1]
                elif self.current_symbol == ".......": # Clear
                    self.decoded_text = ""
                    self.morse_code = ""
                else:
                    ch = self.MORSE_DICT.get(self.current_symbol, "")
                    if ch:
                        self.decoded_text += ch
                
                self.alert_message = self.check_trigger(self.decoded_text)
                self.update_suggestions()
                self.current_symbol = ""
                self.last_blink_end = current_time

           # if not self.current_symbol and pause > self.SPACE_PAUSE:
            #    if self.decoded_text and not self.decoded_text.endswith(" "):
            #        self.decoded_text += " "
            #    self.last_blink_end = current_time

            # Draw the UI
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
            cv2.putText(frame, f"Text: {self.decoded_text}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Morse: {self.morse_code}{self.current_symbol}", (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            
            # Predictions
            cv2.rectangle(frame, (0, h - 130), (300, h), (0, 0, 0), -1)
            for i, s in enumerate(self.suggestions[:3]):
                cv2.putText(frame, f"{i+4}. {s}", (30, h - 70 + i * 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

if __name__ == "__main__":
    # Integration test
    predictor = gemin_predictor.GeminiWordPredictor()
    predictor.load_or_train()
    cam = AutoMorseCam(predictor)
    # This would usually be run by Flask
    print("System Standalone test ready. Use Flask to view.")
