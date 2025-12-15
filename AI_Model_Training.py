from ultralytics import YOLO

# CPS843 Fall 2025 - Wathaned Ean, Assad Kamal, Ivan Wang
# Model training and export using YOLO

def main():

  model = YOLO('yolo12n.pt')

  # train the model
  results = model.train(
    data='datasets/pothole.yaml',
    epochs=50, 
    imgsz=640,
    device="0",
    name="yolov12n_pothole_custom",
  )

  # evaluate model performance on the validation set
  metrics = model.val()

  # perform object detection on an image
  results = model("datasets/train/images/img-383_jpg.rf.abedcd4fce7e8e17889e1bd4274447ba.jpg")
  results[0].show()

  model.export(half=True)

if __name__ == '__main__':
    main()