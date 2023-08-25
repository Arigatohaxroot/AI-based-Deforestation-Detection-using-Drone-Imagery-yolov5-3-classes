import cv2
import torch
from tqdm import tqdm

# Load YOLOv5 model
path = 'C:/Users/user/Downloads/best (6).pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path, force_reload=True)

# Function to perform inference and process video
def process_video(input_video_path, output_directory):
    cap = cv2.VideoCapture(input_video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    filename = input_video_path.split("/")[-1]
    output_video_path = f"{output_directory}/{filename.split('.')[0]}.avi"
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    progress_bar = tqdm(total=total_frames, desc="Processing Frames", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform inference
        results = model(frame)

        # Get annotated frame with bounding boxes
        annotated_frame = results.render()[0]

        # Write the processed frame to output video
        out.write(annotated_frame)

        progress_bar.update(1)  # Update progress bar

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    progress_bar.close()  # Close progress bar

# Input video path and output directory
input_video_path = 'E:/test12.mp4'
output_directory = 'E:/'

# Call the function to process video and save output
process_video(input_video_path, output_directory)
