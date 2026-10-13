"""A closed-loop LIBERO environment, wrapped the way mini-vla wraps Meta-World.

    reset(task_id, start_index) -> (images, state, instruction)
    step(action)                -> (images, state, reward, success, done)
    close()

Three things worth knowing before using it:

* **The images are rotated 180 degrees** (flipped in both axes). LIBERO's renderer produces
  frames rotated relative to the ones stored in the `lerobot/libero` dataset, so a policy
  trained on that dataset would otherwise be shown a flipped scene. Measured on the same
  task and start state, rotating brings the mean pixel gap to the dataset's frame down to
  8.1 for the main camera and 15.0 for the wrist camera, from 73.8 and 81.0 unrotated.
  lerobot's own LIBERO processor applies the same flip.
* **Success comes from check_success(), not from `done`.** `done` is set both when the task
  is solved and when the step limit is reached, so the two are reported separately and the
  caller decides what to do with each.
* The action is 7 numbers in [-1, 1]: three for the hand's position, three for its rotation,
  one for the gripper. The state is 8 numbers - hand position, the hand's rotation as an
  axis-angle vector, and the two gripper joint positions - which is the layout the dataset
  stores, built the same way.

The underlying simulator is one object per task, so asking for a different task rebuilds it.
"""

import os

# Rendering without a display needs EGL. setdefault keeps any value you set yourself.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
import torch
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from robosuite.utils.transform_utils import quat2axisangle

# 3 position, 3 rotation, 1 gripper
DO_NOTHING_ACTION = [0, 0, 0, 0, 0, 0, -1]

# Step the scene a few times after placing it, before the policy acts. LIBERO's own
# evaluation protocol does this, and the demonstrations were recorded the same way, so
# skipping it would show the policy a slightly different scene.
SETTLE_STEPS = 10

# LIBERO's own episode limits, in environment steps
SUITE_MAX_STEPS = {
    "libero_spatial": 280,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def max_steps(suite_name):
    """How many environment steps an episode of this suite may run for."""
    return SUITE_MAX_STEPS[suite_name]


class LiberoEnvironment:
    """One LIBERO task at a time, with the observations shaped the way the policy wants."""

    def __init__(self, suite_name, camera_size):
        self.suite_name = suite_name
        self.suite = benchmark.get_benchmark_dict()[suite_name]()
        self.camera_size = camera_size
        self.task_id = None
        self.env = None

    def reset(self, task_id, start_index):
        """Start `task_id` from one of its 50 fixed start states.

        Returns (images, state, instruction).
        """
        if self.task_id != task_id:
            self.close()
            self.env = self._make_env(task_id)
            self.task_id = task_id

        self.env.seed(0)
        self.env.reset()
        start_states = self.suite.get_task_init_states(task_id)
        # a copy, because the simulator is free to modify the state it is handed and these
        # arrays are reused for every episode of the task
        self.env.set_init_state(np.array(start_states[start_index], copy=True))

        observation = self._settle()
        instruction = self.suite.get_task(task_id).language
        return self.images(observation), self.state(observation), instruction

    def step(self, action):
        """Returns (images, state, reward, success, done)."""
        observation, reward, done, info = self.env.step(action)
        success = bool(self.env.check_success())
        return self.images(observation), self.state(observation), reward, success, done

    def images(self, observation):
        """Both cameras as (2, 3, S, S) float in [0, 1], which is what the backbone wants."""
        main_camera = self._one_image(observation["agentview_image"])
        wrist_camera = self._one_image(observation["robot0_eye_in_hand_image"])
        return torch.stack([main_camera, wrist_camera])

    def state(self, observation):
        """The 8 state numbers, laid out exactly as the dataset stores them."""
        position = observation["robot0_eef_pos"]
        rotation = quat2axisangle(observation["robot0_eef_quat"])
        gripper = observation["robot0_gripper_qpos"]
        joined = np.concatenate([position, rotation, gripper]).astype(np.float32)
        return torch.from_numpy(joined)

    def close(self):
        if self.env is not None:
            self.env.close()
            self.env = None
            self.task_id = None

    def task_count(self):
        """How many tasks this suite has, so callers can loop over all of them."""
        return self.suite.n_tasks

    def _make_env(self, task_id):
        task = self.suite.get_task(task_id)
        bddl_path = get_libero_path("bddl_files")
        bddl_file = os.path.join(bddl_path, task.problem_folder, task.bddl_file)
        return OffScreenRenderEnv(
            bddl_file_name=bddl_file,
            camera_heights=self.camera_size,
            camera_widths=self.camera_size,
        )

    def _settle(self):
        observation = None
        for _ in range(SETTLE_STEPS):
            observation, reward, done, info = self.env.step(DO_NOTHING_ACTION)
        return observation

    def _one_image(self, frame):
        """One camera frame -> (3, S, S) float in [0, 1], rotated and resized.

        The rotation is applied before the resize so the two can never disagree about
        orientation. `ascontiguousarray` is needed because the flip returns a
        negative-stride view and the resize wants a normal one.
        """
        rotated = np.ascontiguousarray(frame[::-1, ::-1])       # 180 degrees
        resized = cv2.resize(rotated, (self.camera_size, self.camera_size),
                             interpolation=cv2.INTER_AREA)
        scaled = resized.astype(np.float32) / 255.0
        return torch.from_numpy(scaled).permute(2, 0, 1)
