"""Process video frames into a video file and encode it to base64."""

import base64
import os

import cv2
import numpy as np


# =====================================================================
def process_video(frames, fps: int) -> str:
    """Convert a list of PIL Image frames into a base64 encoded MP4 video.

    Args:
    ----
        frames (list): A list of PIL Image frames to be converted into a video.
        fps (int): Frames per second for the output video.

    Returns:
    -------
        str: A base64 encoded string representing the MP4 video.

    """
    if not frames:
        raise ValueError("No frames provided")

    # Convert first frame to get size
    first_frame = np.array(frames[0])
    height, width, _ = first_frame.shape

    # Use BytesIO to capture video in memory
    temp_path = "temp_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    for frame in frames:
        np_frame = np.array(frame)
        bgr_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr_frame)

    writer.release()

    # Read back as bytes
    with open(temp_path, "rb") as f:
        video_bytes = f.read()

    os.remove(temp_path)

    # Encode to base64
    video_base64 = base64.b64encode(video_bytes).decode("utf-8")
    return video_base64
