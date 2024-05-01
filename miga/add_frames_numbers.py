import cv2
import os.path as osp
from tqdm import tqdm

def add_frame_number_to_video(input_video_path, output_video_path):
    input_video_path = osp.join(*input_video_path.split('/'))
    output_video_path = osp.join(*output_video_path.split('/'))

    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':

    # input_video_path = "D:\\Project-mpg microgesture\\imigue\\0001.mp4"
    # output_video_path = "D:\\Project-mpg microgesture\\imigue\\0001_frame_number.mp4"

    input_video_path = 'D:\Project-mpg microgesture\smg\Sample0031_color.mp4'
    output_video_path = 'D:\Project-mpg microgesture\smg\Sample0031_color_frame_number.mp4'

    add_frame_number_to_video(input_video_path, output_video_path)
