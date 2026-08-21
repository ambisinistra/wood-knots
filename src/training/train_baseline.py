from ultralytics import YOLO


def train_baseline():
    # 1. Загружаем легкую предобученную модель segmentation
    model = YOLO("yolo11s-seg.pt")  # или yolo8s-seg.pt

    # 2. Обучаем baseline
    _ = model.train(
        data="configs/data.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        project="runs/segment",
        name="baseline",
        save=True,
        plots=True,
        seed=42,
    )


if __name__ == "__main__":

    train_baseline()