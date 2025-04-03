from ultralytics import YOLO
import torch
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "Poles", "lidar", "data.yaml")

    img_size = 1024
    epochs = 200
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
        name='exp_lidar',
        patience=50,
        pretrained=True,
        optimizer='Adam',
        lr0=0.001,
        rect=True,
    )

    metrics = model.val()
    print("Evaluation Metrics (LiDAR):")
    print(f"Mean Recall (mr): {metrics.box.mr:.4f}")
    print(f"Mean AP@50 (map50): {metrics.box.map50:.4f}")
    print(f"Mean AP@50:0.95 (map): {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()