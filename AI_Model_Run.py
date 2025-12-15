from ultralytics import YOLO
import cv2 as cv2

# CPS843 Fall 2025 - Wathaned Ean, Assad Kamal, Ivan Wang

def run_model(model, video, output_video):
    model = model
    cap = cv2.VideoCapture(video)

    # create videoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # get frame width and height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video, fourcc, 20.0, (frame_width, frame_height))

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        # capture frame by frame
        ret, frame = cap.read()

        if not ret:
            print("No frame...")
            break

        # predict on image
        results = model.track(source=frame, conf=0.30, persist=True, tracker='bytetrack.yaml')
        frame = results[0].plot()
        frame = cv2.resize(frame, (1280, 720))

        # write the frame to the output video file
        out.write(frame)

        # display the resulting frame
        cv2.imshow("ObjectDetection", frame)

        # terminate run when "Q" pressed
        if cv2.waitKey(1) == ord("q"):
            break

    # when everything done release the capture
    cap.release()

    # release the video recording
    # out.release()
    cv2.destroyAllWindows()

# object detection

run_model(model=YOLO('best.pt', "v12"), video=VIDEO_FILEPATH, output_video=OUTPUT_VIDEO_FILEPATH)
