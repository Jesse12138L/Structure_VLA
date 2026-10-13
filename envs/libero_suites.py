"""Standalone viewer: look at a LIBERO scene before using it for anything.

This is not part of any training or evaluation pipeline. It does three things:

  1. prints every task of the five LIBERO suites, with its language instruction
  2. prints what the environment gives you: observation fields and the action space
  3. shows one scene with the arm moving (small random actions, there is no expert)

Run it from the project root, inside the `lerobot` conda env:

    python -m envs.libero_suites

Change the constants below to look at a different task. With SHOW_WINDOW = True it opens
a window (needs a display, press ESC to quit). With SHOW_WINDOW = False it writes a
video to VIDEO_PATH instead, which is what you want on a machine without a display.
"""

import os

# Rendering without a display needs EGL. setdefault keeps any value you set yourself.
os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# libero_spatial / libero_object / libero_goal / libero_10 / libero_90
TASK_SUITE = "libero_10"
TASK_ID = 5
CAMERA_SIZE = 256        # both cameras are rendered as CAMERA_SIZE x CAMERA_SIZE
NUM_STEPS = 200
SHOW_WINDOW = True
WINDOW_SCALE = 2         # the window is the camera image times this
VIDEO_PATH = "videos/libero_view.mp4"
VIDEO_FPS = 20           # LIBERO controls the robot at 20 Hz

SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]

# Seven numbers: 3 for the hand position, 3 for its rotation, 1 for the gripper.
# This action does nothing and keeps the gripper open (-1 is open, +1 is closed).
DO_NOTHING_ACTION = [1, 2, 3, 0, 0, 0, 1]
SETTLE_STEPS = 10        # let the objects stop moving after the scene is set up
RANDOM_ACTION_SIZE = 0.3


def print_task_catalog():
    """Print every task of every suite, with the instruction a policy would be given."""
    benchmark_dict = benchmark.get_benchmark_dict()
    for suite_name in SUITE_NAMES:
        suite = benchmark_dict[suite_name]()
        print("")
        print(suite_name + ": " + str(suite.n_tasks) + " tasks")
        for task_id in range(suite.n_tasks):
            task = suite.get_task(task_id)
            print("  " + str(task_id).rjust(2) + "  " + task.language)


def make_env(task):
    """Build the simulator for one task. A task is a scene file (.bddl) plus a goal."""
    bddl_folder = get_libero_path("bddl_files")
    bddl_path = os.path.join(bddl_folder, task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_path,
        camera_heights=CAMERA_SIZE,
        camera_widths=CAMERA_SIZE,
    )
    return env


def reset_to_first_start(env, suite):
    """Reset, then put the scene in the first of the task's 50 fixed start states.

    LIBERO ships 50 start states per task, so every run can begin from the same scene.
    """
    env.seed(0)
    env.reset()
    start_states = suite.get_task_init_states(TASK_ID)
    env.set_init_state(start_states[0])

    obs = None
    for _ in range(SETTLE_STEPS):
        obs, reward, done, info = env.step(DO_NOTHING_ACTION)
    return obs


def print_observation_and_actions(env, obs):
    """Show what the environment returns, so the shapes are not a guess."""
    print("")
    print("observation fields:")
    for name in obs:
        value = np.asarray(obs[name])
        print("  " + name.ljust(40) + str(value.dtype).ljust(10) + str(value.shape))

    action_low, action_high = env.env.action_spec
    print("")
    print("action space: " + str(len(action_low)) + " numbers")
    print("  low : " + str(action_low))
    print("  high: " + str(action_high))


def make_frame(obs):
    """Put the two cameras side by side, in the BGR order that OpenCV expects."""
    main_camera = obs["agentview_image"]
    wrist_camera = obs["robot0_eye_in_hand_image"]
    both = np.concatenate([main_camera, wrist_camera], axis=1)
    frame = cv2.cvtColor(both, cv2.COLOR_RGB2BGR)
    return frame


def view(env, obs):
    """Move the arm with small random actions and show or record every frame."""
    random_generator = np.random.default_rng(0)

    video_writer = None
    if not SHOW_WINDOW:
        os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)
        frame = make_frame(obs)
        height, width = frame.shape[0], frame.shape[1]
        codec = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(VIDEO_PATH, codec, VIDEO_FPS, (width, height))

    try:
        for step in range(NUM_STEPS):
            action = random_generator.uniform(-RANDOM_ACTION_SIZE, RANDOM_ACTION_SIZE, size=7)
            obs, reward, done, info = env.step(action)
            frame = make_frame(obs)

            if SHOW_WINDOW:
                big_frame = cv2.resize(frame, None, fx=WINDOW_SCALE, fy=WINDOW_SCALE,
                                       interpolation=cv2.INTER_NEAREST)
                cv2.imshow("LIBERO " + TASK_SUITE + " task " + str(TASK_ID), big_frame)
                key = cv2.waitKey(30)
                if key == 27:  # ESC
                    break
            else:
                video_writer.write(frame)

            # LIBERO only sets done when the task is solved, which random actions will not do
            if done:
                break
    finally:
        if video_writer is not None:
            video_writer.release()
            print("")
            print("saved video to " + VIDEO_PATH)
        cv2.destroyAllWindows()


def main():
    print_task_catalog()

    suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
    task = suite.get_task(TASK_ID)
    print("")
    print("viewing " + TASK_SUITE + " task " + str(TASK_ID) + ": " + task.language)

    env = make_env(task)
    try:
        obs = reset_to_first_start(env, suite)
        print_observation_and_actions(env, obs)
        view(env, obs)
    finally:
        env.close()


if __name__ == "__main__":
    main()
