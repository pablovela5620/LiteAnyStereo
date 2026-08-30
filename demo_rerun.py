"""LiteAnyStereo V2 on one calibrated stereo pair, visualized in Rerun.

Same preprocessing as ``demo.py`` (float 0-255 NCHW, InputPadder divisible by 32);
the only additions are the exoego:v2 rig layout, metric depth (fx * baseline / disparity)
under the left pinhole so the viewer backprojects it, and GT disparity + error when the
Middlebury-style ``disp0GT.pfm`` / ``mask0nocc.png`` sit beside the pair.

    pixi run demo                # ETH3D playground_1l, LAS2-M
    pixi run demo --model-size h # LAS2-H
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from PIL import Image
from simplecv.camera_parameters import Extrinsics, Intrinsics, PinholeParameters
from simplecv.rerun_log_utils import RerunTyroConfig
from simplecv.rerun_rig_logger import log_rig_static
from simplecv.rig import CameraSensor, Rig, RigCalibration
import tyro

from core.models import build_model, load_model_weights
from core.utils.frame_utils import readPFM
from core.utils.utils import InputPadder


@dataclass
class Config:
    rr_config: RerunTyroConfig
    scene_dir: Path = Path("data/datasets/ETH3D/two_view_training/playground_1l")
    """Directory with im0.png, im1.png and a Middlebury-style calib.txt."""
    gt_dir: Path | None = None
    """Directory with disp0GT.pfm + mask0nocc.png; defaults to the two_view_training_gt twin of scene_dir."""
    model_size: Literal["s", "m", "l", "h"] = "m"
    checkpoint: Path | None = None
    """Defaults to checkpoints/LAS2_<SIZE>.pth."""
    max_disp: int = 192
    max_depth_m: float = 20.0
    """Depth beyond this is dropped from the backprojected cloud (sub-pixel disparities explode)."""
    device: Literal["cuda", "cpu"] = "cuda"


def read_calib(path: Path) -> tuple[np.ndarray, float, float]:
    """Middlebury v3 calib.txt -> (K of cam0, baseline in metres, doffs in px)."""
    text = path.read_text()
    cam0 = re.search(r"cam0=\[(.*?)\]", text).group(1)
    K = np.array([[float(v) for v in row.split()] for row in cam0.split(";")])
    baseline_mm = float(re.search(r"baseline=([\d.]+)", text).group(1))
    doffs = float(re.search(r"doffs=([\d.]+)", text).group(1))
    return K, baseline_mm / 1000.0, doffs


def main(cfg: Config) -> None:
    left = np.asarray(Image.open(cfg.scene_dir / "im0.png").convert("RGB"))
    right = np.asarray(Image.open(cfg.scene_dir / "im1.png").convert("RGB"))
    K, baseline_m, doffs = read_calib(cfg.scene_dir / "calib.txt")
    H, W = left.shape[:2]

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    checkpoint = cfg.checkpoint or Path(f"checkpoints/LAS2_{cfg.model_size.upper()}.pth")
    model = build_model("las2", fnet_pretrained=False, model_size=cfg.model_size, max_disp=cfg.max_disp)
    load_model_weights(model, torch.load(checkpoint, map_location=device), strict=True)
    model = model.to(device).eval()

    img0_t = torch.as_tensor(left, device=device).float()[None].permute(0, 3, 1, 2)
    img1_t = torch.as_tensor(right, device=device).float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(img0_t.shape, divis_by=32)
    img0_t, img1_t = padder.pad(img0_t, img1_t)
    with torch.no_grad():
        disp_t = model(img0_t, img1_t, max_disp=cfg.max_disp, test_mode=True)
    disp = padder.unpad(disp_t.float()).cpu().numpy().reshape(H, W).astype(np.float32)

    # ── exoego:v2 rig: cam_00 = left (reference), cam_01 = right at +baseline along x ──
    def sensor(index: int, name: str, t_x: float) -> CameraSensor:
        intrinsics = Intrinsics(camera_conventions="RDF", height=H, width=W, k_matrix=K)
        extrinsics = Extrinsics(world_R_cam=np.eye(3), world_t_cam=np.array([t_x, 0.0, 0.0]))
        return CameraSensor(index=index, name=name, kind="rgb", pinhole=PinholeParameters(name=name, extrinsics=extrinsics, intrinsics=intrinsics))

    rig = Rig(index=0, calibration=RigCalibration(cameras=[sensor(0, "left", 0.0), sensor(1, "right", baseline_m)]), image_plane_distance=0.5)
    rr.log("/", rr.ViewCoordinates.RDF, static=True)
    log_rig_static(rig)
    left_path, right_path = "world/rig_00/cam_00", "world/rig_00/cam_01"
    rr.log(f"{left_path}/pinhole/image", rr.Image(left).compress(jpeg_quality=90), static=True)
    rr.log(f"{right_path}/pinhole/image", rr.Image(right).compress(jpeg_quality=90), static=True)

    # Metric depth from disparity; sub-pixel disparities (sky) explode to km, so keep only depth <= max_depth_m.
    depth = np.zeros_like(disp)
    valid = disp > 0.0
    depth[valid] = K[0, 0] * baseline_m / (disp[valid] + doffs)
    depth[depth > cfg.max_depth_m] = 0.0
    rr.log(f"{left_path}/pinhole/depth", rr.DepthImage(depth, meter=1.0, depth_range=(0.0, cfg.max_depth_m)), static=True)
    rr.log(f"{left_path}/disparity", rr.DepthImage(disp), static=True)

    gt_dir = cfg.gt_dir or Path(str(cfg.scene_dir).replace("two_view_training", "two_view_training_gt"))
    tabs = [rrb.Spatial2DView(origin=f"{left_path}/disparity", name="disparity")]
    if (gt_dir / "disp0GT.pfm").exists():
        gt = np.ascontiguousarray(readPFM(str(gt_dir / "disp0GT.pfm"))).astype(np.float32)
        nocc = np.asarray(Image.open(gt_dir / "mask0nocc.png")) == 255
        gt_valid = np.isfinite(gt) & (gt < cfg.max_disp) & nocc
        err = np.abs(disp - gt)
        epe = float(err[gt_valid].mean())
        bad1 = 100.0 * float((err[gt_valid] > 1.0).mean())
        rr.log(f"{left_path}/disparity_gt", rr.DepthImage(np.where(np.isfinite(gt), gt, 0.0)), static=True)
        rr.log(f"{left_path}/disparity_error", rr.DepthImage(np.where(gt_valid, err, 0.0), depth_range=(0.0, 5.0)), static=True)
        rr.log("metrics", rr.TextDocument(f"{cfg.scene_dir.name} LAS2-{cfg.model_size.upper()}\nEPE {epe:.3f} px\nbad1 {bad1:.2f} % (non-occluded, gt < {cfg.max_disp})", media_type=rr.MediaType.MARKDOWN), static=True)
        print(f"{cfg.scene_dir.name} LAS2-{cfg.model_size.upper()}: EPE {epe:.3f}  bad1 {bad1:.2f}%")
        tabs += [rrb.Spatial2DView(origin=f"{left_path}/disparity_gt", name="GT"), rrb.Spatial2DView(origin=f"{left_path}/disparity_error", name="|error| px")]

    rr.send_blueprint(
        rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world", contents=["$origin/**", f"- {left_path}/pinhole/image", f"- {right_path}/pinhole/image", f"- {left_path}/disparity", f"- {left_path}/disparity_gt", f"- {left_path}/disparity_error"]),
                rrb.Vertical(
                    rrb.Horizontal(rrb.Spatial2DView(origin=f"{left_path}/pinhole/image", name="left"), rrb.Spatial2DView(origin=f"{right_path}/pinhole/image", name="right")),
                    rrb.Spatial2DView(origin=f"{left_path}/pinhole/depth", name="depth (m)"),
                    rrb.Tabs(*tabs, active_tab=0),
                ),
                column_shares=[3, 2],
            ),
            collapse_panels=True,
        )
    )


if __name__ == "__main__":
    main(tyro.cli(Config))
