# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model

from rlinf.envs.behavior.control_modes import (
    extract_state_from_proprio,
)


def make_behavior_example() -> dict:
    """Creates a random input example for the Behavior policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    elif image.shape[0] == 2 and image.shape[1] == 3:
        image = einops.rearrange(image, "n c h w -> n h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BehaviorInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.
    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    model_type: _model.ModelType
    extract_state_from_proprio: bool = False
    use_all_wrist_images: bool = False
    state_token: str = "abs_joint_old"

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right).
        # If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and
        # replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation/image"])  # [h, w, c]
        wrist_image = _parse_image(
            data["observation/wrist_image"]
        )  # [num_image, h, w, c]

        state = (
            extract_state_from_proprio(
                data["observation/state"], state_token=self.state_token
            )
            if self.extract_state_from_proprio
            else data["observation/state"]
        )

        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": np.asarray(state)[..., :32],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image[0, ...],
                "right_wrist_0_rgb": wrist_image[1, ...],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_
                if self.model_type == _model.ModelType.PI0_FAST
                or self.use_all_wrist_images
                else np.False_,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            inputs["actions"] = data["actions"]

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # Full π₀.₅ SFT supervises an autoregressive subtask response. Keep it
        # alongside the main-task prompt until the model-transform tokenizer
        # consumes both fields. Action-only VLA samples simply omit this key.
        if "response" in data:
            inputs["response"] = data["response"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.
    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    action_dim: int = 23

    def __call__(self, data: dict) -> dict:
        # Return only the selected BEHAVIOR representation. Model-space actions
        # are padded to 32 channels, while joint and EEF controllers consume 23
        # and 21 channels, respectively.
        return {"actions": np.asarray(data["actions"][:, : self.action_dim])}
