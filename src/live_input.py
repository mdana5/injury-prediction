import cv2
import time

def record_video(duration, output_file="input_video.mp4"):
    cap = cv2.VideoCapture(0)  # 0 = webcam
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    start_time = time.time()
    print("Recording started...")

    while int(time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            print("❌ Error: Failed to read frame")
            break

        out.write(frame)
        cv2.imshow("Recording...", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # press Q to quit early
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Recording finished. Saved as:", output_file)

# CALL THE FUNCTION HERE
record_video(duration=5)
