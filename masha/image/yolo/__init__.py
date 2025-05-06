from pathlib import Path
from ultralytics.solutions import ObjectCounter
import os
import cv2
from coreimage.terminal import print_term_image

HGROOT = Path(os.environ.get("HUGGINGROOT"))


def detect_objects(img_path: Path):
    # image = load_image(img_path.as_posix())
    # model_path = HGROOT / "yolo/yolo11x.pt"
    
    # model = YOLO(model_path)
    # res = model.predict(source=img_path)
    # for r in res:
    #     print(r.boxes)
        
    
    segmenter = ObjectCounter()
    frame = cv2.imread(img_path.as_posix())
    results = segmenter.process(frame)
    print_term_image(image=results.plot_im, height=20)

    