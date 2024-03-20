from typing import Optional, Dict, Literal, Set

import numpy as np

import viser
import viser.theme
import viser.transforms as vtf

from nerfstudio.viewer.viewer import Viewer
from nerfstudio.data.datasets.base_dataset import InputDataset

VISER_NERFSTUDIO_SCALE_RATIO: float = 10.0


class ROSViewer(Viewer):
    """
    A viewer that supports rendering streaming training views.
    """

    def init_scene(
        self,
        train_dataset: InputDataset,
        train_state: Literal["training", "paused", "completed"],
        eval_dataset: Optional[InputDataset] = None,
    ) -> None:
        """Draw some images and the scene aabb in the viewer.

        Args:
            dataset: dataset to render in the scene
            train_state: Current status of training
        """
        # draw the training cameras and images
        self.camera_handles: Dict[int, viser.CameraFrustumHandle] = {}
        self.original_c2w: Dict[int, np.ndarray] = {}
        self.cameras_drawn: Set[int] = set()
        self.dataset = train_dataset
        image_indices = self._pick_drawn_image_idxs(len(train_dataset))
        cameras = train_dataset.cameras.to("cpu")
        for idx in image_indices:
            camera = cameras[idx]
            c2w = camera.camera_to_worlds.cpu().numpy()
            R = vtf.SO3.from_matrix(c2w[:3, :3])
            R = R @ vtf.SO3.from_x_radians(np.pi)
            camera_handle = self.viser_server.add_camera_frustum(
                name=f"/cameras/camera_{idx:05d}",
                fov=float(2 * np.arctan(camera.cx / camera.fx[0])),
                scale=self.config.camera_frustum_scale,
                aspect=float(camera.cx[0] / camera.cy[0]),
                image=None,
                wxyz=R.wxyz,
                position=c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO,
                visible=False,
            )

            @camera_handle.on_click
            def _(
                event: viser.SceneNodePointerEvent[viser.CameraFrustumHandle],
            ) -> None:
                with event.client.atomic():
                    event.client.camera.position = event.target.position
                    event.client.camera.wxyz = event.target.wxyz

            self.camera_handles[idx] = camera_handle
            self.original_c2w[idx] = c2w

        self.train_state = train_state
        self.train_util = 0.9

    def update_camera_poses(self):
        """Updates the camera poses in the scene."""
        image_indices = self.dataset.updated_indices
        for idx in image_indices:
            if (not idx in self.cameras_drawn) and (idx in self.camera_handles):
                self.original_c2w[idx] = (
                    self.dataset.cameras.camera_to_worlds[idx].cpu().numpy()
                )
                self.camera_handles[idx].visible = True
                self.cameras_drawn.add(idx)

                # TODO: This is a hack to render cameras because gsplat does
                # not support camera optimization. Remove this when gsplat is fixed.
                if not hasattr(self.pipeline.model, "camera_optimizer"):
                    c2w = self.original_c2w[idx]

                    R = vtf.SO3.from_matrix(c2w[:3, :3])  # type: ignore
                    R = R @ vtf.SO3.from_x_radians(np.pi)
                    self.camera_handles[idx].position = (
                        c2w[:3, 3] * VISER_NERFSTUDIO_SCALE_RATIO
                    )
                    self.camera_handles[idx].wxyz = R.wxyz

        super().update_camera_poses()
