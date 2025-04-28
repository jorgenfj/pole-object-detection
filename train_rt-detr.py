from ultralytics import RTDETR
import torch
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    yaml_path = os.path.join(current_dir, "Poles", "rgb", "data.yaml")

    img_size = 640
    epochs = 200
    batch_size = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load RT-DETR model (instead of YOLO)
    model = RTDETR("rtdetr-l.pt")

    model.info()

    model.train(
    data=yaml_path,
    epochs=epochs,
    imgsz=img_size,
    batch=batch_size,
    device=device,
    project='runs/train',
    name='exp_rgb_rtdetr',
    patience=50,
    pretrained=True,
    optimizer='AdamW',        # Use AdamW, better for transformers
    lr0=5e-4,                 # Lower starting LR: 0.0005 (instead of 0.001)
    weight_decay=0.05,        # Higher weight decay is recommended
    rect=True,
    warmup_epochs=5,          # Slightly longer warmup (more stable start)
)

    metrics = model.val()
    print("Evaluation Metrics (LiDAR):")
    print(f"Mean Recall (mr): {metrics.box.mr:.4f}")
    print(f"Mean AP@50 (map50): {metrics.box.map50:.4f}")
    print(f"Mean AP@50:0.95 (map): {metrics.box.map:.4f}")

if __name__ == "__main__":
    main()
