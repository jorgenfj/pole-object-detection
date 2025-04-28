from ultralytics import YOLO
import torch
import os

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your trained model weights
    model_path = os.path.join(current_dir, "runs", "train", "exp_rgb6", "weights", "best.pt")
    
    # Path to your test images
    test_images_path = os.path.join(current_dir, "Poles", "rgb", "images", "test")
    
    # Where to save the predictions
    save_project = os.path.join(current_dir, "runs", "predict")
    save_name = "exp_rgb_yolon_test"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the trained model
    model = YOLO(model_path)

    # Predict on the test set
    model.predict(
        source=test_images_path,
        imgsz=640,                  # match your training imgsz
        device=device,
        project=save_project,
        name=save_name,
        save_txt=True,               # save raw predictions as YOLO .txt files
        save_conf=True,              # include confidence scores in .txt files
    )

    print(f"Predictions saved to {os.path.join(save_project, save_name)}")

if __name__ == "__main__":
    main()
