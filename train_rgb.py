
from ultralytics import YOLO
import torch
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "Poles", "rgb", "data.yaml")

    img_size = 640
    epochs = 1
    batch_size = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO('yolov8s.pt')

    model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project='runs/train',
        name='exp_rgb',
        patience=50,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
    )

    metrics = model.val()
    print("Evaluation Metrics (RGB Only):")
    print(f"Mean Recall (mr): {metrics.box.mr:.4f}")
    print(f"Mean AP@50 (map50): {metrics.box.map50:.4f}")
    print(f"Mean AP@50:0.95 (map): {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
