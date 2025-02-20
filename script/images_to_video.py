import cv2
import os

def images_to_video(image_folder, output_video, fps=30):
    images = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])
    
    if not images:
        print("No images found in the folder.")
        return

    first_image = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = first_image.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for image in images:
        frame = cv2.imread(os.path.join(image_folder, image))
        video_writer.write(frame)

    video_writer.release()
    print(f"Video saved as {output_video}")

# Example usage
images_to_video("datasets/video/train/images", "datasets/video/sample.mp4")
