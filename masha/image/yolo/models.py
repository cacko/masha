from pathlib import Path
from typing import Optional
from corestring import titlecase
from pydantic import BaseModel
from pathlib import Path
import numpy as np
from ultralytics.engine.results import Results as UltraResults
from copy import deepcopy
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.torch_utils import smart_inference_mode
import cv2

from masha.image.classifiers.models import OBJECT


class ObjectCrop(BaseModel, arbitrary_types_allowed=True):
    path: Path
    cls: str
    idx: int

    @property
    def label(self):
        return f"{titlecase(self.cls)} #{self.idx} "


class CropResults(BaseModel, arbitrary_types_allowed=True):
    plot_im: np.ndarray
    objects: list[ObjectCrop]

    def forClass(self, cls: str):
        return list(filter(lambda x: x.cls == cls, self.objects))
    
    def save(self, filename):
        cv2.imwrite(filename, np.asarray(self.plot_im))


class Results(UltraResults):

    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font=None,
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
        txt_color=(255, 255, 255),
        show_only: Optional[list[OBJECT]] = None,
    ):
        assert color_mode in {
            "instance",
            "class",
        }, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (
                (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255)
                .to(torch.uint8)
                .cpu()
                .numpy()
            )

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width=line_width,
            font_size=font_size,
            font=font,
            pil=pil
            or (
                pred_probs is not None and show_probs
            ),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(
                        img, dtype=torch.float16, device=pred_masks.data.device
                    )
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"
                else (
                    pred_boxes.cls
                    if pred_boxes and color_mode == "class"
                    else reversed(range(len(pred_masks)))
                )
            )
            annotator.masks(
                pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu
            )

        # Plot Detect results
        printed = []
        if pred_boxes is not None and show_boxes:
            found_classes = pred_boxes.cls.tolist()
            for i, d in enumerate(reversed(pred_boxes)):
                c, d_conf, id = (
                    int(d.cls),
                    float(d.conf) if conf else None,
                    None if d.id is None else int(d.id.item()),
                )
                is_muultiple_cls = found_classes.count(c) > 1
                name = names[c]
                if show_only and name not in show_only:
                    continue
                printed.append(name)
                cls_number = f"#{printed.count(name)}" if is_muultiple_cls else ""
                label = f"{titlecase(name)} {cls_number}".strip()
                box = (
                    d.xyxyxyxy.reshape(-1, 4, 2).squeeze()
                    if is_obb
                    else d.xyxy.squeeze()
                )
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        (
                            c
                            if color_mode == "class"
                            else (
                                id
                                if id is not None
                                else i if color_mode == "instance" else None
                            )
                        ),
                        True,
                    ),
                    rotated=is_obb,
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = "\n".join(
                f"{names[j] if names else j} {pred_probs.data[j]:.2f}"
                for j in pred_probs.top5
            )
            x = round(self.orig_shape[0] * 0.03)
            annotator.text(
                [x, x], text, txt_color=txt_color, box_color=(64, 64, 64, 128)
            )  # RGBA box

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename or f"results_{Path(self.path).name}")

        return annotator.im if pil else annotator.result()
