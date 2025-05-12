from typing import Optional
from corefile import TempPath
import numpy as np
from ultralytics.solutions.solutions import BaseSolution
from ultralytics.utils.plotting import save_one_box
from uuid import uuid4
import numpy as np
from ultralytics.utils.plotting import save_one_box

from masha.image.classifiers.models import OBJECT
from .models import Results, ObjectCrop, CropResults, MODELS_PATH


class ObjectCropper(BaseSolution):
    def __init__(self, **kwargs):
        kwargs["model"] = MODELS_PATH / "yolov8x-worldv2.pt"
        kwargs["conf"] = 0.4
        super().__init__(**kwargs)

        self.crop_dir = TempPath("objects-cropped")
        if not self.crop_dir.exists():
            self.crop_dir.mkdir(parents=True)
        if self.CFG["show"]:
            self.LOGGER.warning(
                f"show=True disabled for crop solution, results will be saved in the directory named: {self.crop_dir}"
            )
        self.crop_idx = 0  # Initialize counter for total cropped objects
        self.iou = self.CFG["iou"]
        self.conf = self.CFG["conf"]

    def process(self, im0: np.ndarray, show_only: Optional[list[OBJECT]] = None):
        self.extract_tracks(im0)
        results = Results(
            im0,
            path=None,
            names=self.names,
            boxes=self.track_data.data,
        )

        objects = []
        gen_id = uuid4()
        processed = []
        detected_classes = results.boxes.cls.tolist()
        for box, cls_idx in zip(reversed(results.boxes), reversed(results.boxes.cls)):
            is_multiple_cls = detected_classes.count(cls_idx.item()) > 1
            cls = results.names[cls_idx.item()]
            self.crop_idx += 1
            processed.append(cls)
            crop_path = self.crop_dir / f"{gen_id}_{self.crop_idx}.jpg"
            save_one_box(
                box.xyxy,
                im0,
                file=crop_path,
                BGR=True,
            )
            objects.append(
                ObjectCrop(
                    path=crop_path,
                    idx=(processed.count(cls) if is_multiple_cls else 0),
                    cls=cls,
                )
            )

        # Return SolutionResults
        return CropResults(
            plot_im=results.plot(
                show_only=[],
                line_width=2
            ),
            objects=objects,
        )
