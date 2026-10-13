"""Standalone viewer: look at the LIBERO demonstration data before using it for anything.

This is not part of any training pipeline. The first run downloads the dataset
`lerobot/libero` (about 1.9 GB) into DATA_ROOT, inside this project. Then it prints:

  1. what the dataset contains: episodes, frames, fps, and every feature
  2. which dataset tasks belong to which LIBERO suite (matched by the instruction text)
  3. how long the demonstrations of the focus suite are
  4. what one sample looks like, including a chunk of future actions
  5. a picture of a few frames, saved to PICTURE_PATH

Run it from the project root, inside the `lerobot` conda env:

    python -m data.libero_data

If the download fails with "Unknown scheme for proxy URL", your shell has a socks:// proxy
that the download library cannot parse. Run `unset ALL_PROXY all_proxy` first.
"""

import cv2
import numpy as np
from libero.libero import benchmark
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

#HF 上的仓库名 + 本地存放路径。两个一起传给 LeRobotDataset，效果是"本地有就用本地，没有就去下载"
REPO_ID = "lerobot/libero"
DATA_ROOT = "data/lerobot_libero"
PICTURE_PATH = "data/libero_frames.png"

# libero_90 is not in this dataset, it only has these four suites (40 tasks)
SUITE_NAMES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
FOCUS_SUITE = "libero_goal"
FOCUS_STEP_LIMIT = 300   # the longest an episode may run when this suite is evaluated

CHUNK_SIZE = 10          # how many future actions to ask for with every sample
PICTURE_EPISODES = 2     # how many episodes to draw in the picture

MAIN_CAMERA_KEY = "observation.images.image"
WRIST_CAMERA_KEY = "observation.images.image2"


def print_overview(meta):
    """Print the size of the dataset and every feature it stores."""
    print("episodes: " + str(meta.total_episodes))
    print("frames  : " + str(meta.total_frames))
    print("fps     : " + str(meta.fps))

    print("")
    print("features:")
    for name in meta.features:
        spec = meta.features[name]
        print("  " + name.ljust(28) + str(spec["dtype"]).ljust(9) + str(spec["shape"]))

    # The dataset only calls the state "state", it does not say which 8 numbers they are
    for name in ["observation.state", "action"]:
        mean = np.round(np.asarray(meta.stats[name]["mean"]), 3)
        std = np.round(np.asarray(meta.stats[name]["std"]), 3)
        print("")
        print(name + " mean: " + str(mean))
        print(name + " std : " + str(std))


def match_tasks_to_suites(meta):
    """Find which dataset task is which LIBERO task, by comparing the instruction text.

    Returns a dict: suite name -> list of dataset task indices, in LIBERO task order.
    """
    task_index_of_language = meta.tasks["task_index"].to_dict()
    benchmark_dict = benchmark.get_benchmark_dict()

    task_indices_of_suite = {}
    print("")
    print("tasks per suite:")
    for suite_name in SUITE_NAMES:
        suite = benchmark_dict[suite_name]()
        found = []
        missing = []
        for task_id in range(suite.n_tasks):
            language = suite.get_task(task_id).language
            if language in task_index_of_language:
                found.append(task_index_of_language[language])
            else:
                missing.append(language)

        task_indices_of_suite[suite_name] = found
        print("  " + suite_name.ljust(16) + str(len(found)) + " of " + str(suite.n_tasks)
              + " matched, dataset task indices " + str(found))
        for language in missing:
            print("    NOT FOUND: " + language)
    return task_indices_of_suite


def group_episodes_by_task(dataset, meta):
    """Return a dict: dataset task index -> list of episodes (each one a dict).

    The episode table does not say which task an episode is, so look at the task of its
    first frame.
    """
    episodes_of_task = {}
    for episode in meta.episodes:
        first_frame = episode["dataset_from_index"]
        row = dataset.hf_dataset[first_frame]
        task_index = int(row["task_index"])
        if task_index not in episodes_of_task:
            episodes_of_task[task_index] = []
        episodes_of_task[task_index].append(episode)
    return episodes_of_task


def print_episode_lengths(meta, task_indices, episodes_of_task):
    """Print how many frames the demonstrations of the focus suite have."""
    print("")
    print(FOCUS_SUITE + " demonstration lengths in frames (evaluation step limit is "
          + str(FOCUS_STEP_LIMIT) + "):")
    for task_index in task_indices:
        episodes = episodes_of_task[task_index]
        lengths = []
        for episode in episodes:
            lengths.append(episode["length"])
        average = sum(lengths) / len(lengths)
        language = meta.tasks.index[task_index]
        print("  task " + str(task_index).rjust(2) + "  episodes " + str(len(lengths)).rjust(3)
              + "  min " + str(min(lengths)).rjust(3) + "  mean " + str(round(average)).rjust(3)
              + "  max " + str(max(lengths)).rjust(3) + "   " + language)


def print_sample(sample, title):
    """Print every field of one sample: tensors by shape and type, the rest as they are."""
    print("")
    print(title)
    for key in sample:
        value = sample[key]
        if hasattr(value, "shape"):
            print("  " + key.ljust(28) + str(value.dtype).ljust(16) + str(tuple(value.shape)))
        else:
            print("  " + key.ljust(28) + str(value))

    image = sample[MAIN_CAMERA_KEY]
    print("  main camera value range: " + str(float(image.min())) + " to " + str(float(image.max())))


def picture_from_tensor(image):
    """Turn a (3, H, W) float image in [0, 1] into an (H, W, 3) uint8 image."""
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def draw_picture(dataset, episodes):
    """Save frames from the start, middle and end of some episodes to one picture.

    One column per episode, one row per moment, main camera on the left of each row and
    wrist camera on the right.
    """
    columns = []
    for episode in episodes:
        first = episode["dataset_from_index"]
        last = episode["dataset_to_index"] - 1
        middle = (first + last) // 2

        rows = []
        for frame_index in [first, middle, last]:
            sample = dataset[frame_index]
            main_camera = picture_from_tensor(sample[MAIN_CAMERA_KEY])
            wrist_camera = picture_from_tensor(sample[WRIST_CAMERA_KEY])
            rows.append(np.concatenate([main_camera, wrist_camera], axis=1))
        columns.append(np.concatenate(rows, axis=0))

    picture = np.concatenate(columns, axis=1)
    picture = cv2.cvtColor(picture, cv2.COLOR_RGB2BGR)
    cv2.imwrite(PICTURE_PATH, picture)
    print("")
    print("saved picture to " + PICTURE_PATH + " (columns: episodes, rows: start / middle / end)")


def main():
    meta = LeRobotDatasetMetadata(REPO_ID, root=DATA_ROOT)
    print_overview(meta)
    task_indices_of_suite = match_tasks_to_suites(meta)

    # Asking for the action at several moments gives a chunk of future actions. Times are
    # in seconds, so step 3 is 3 / fps seconds after the current frame.
    action_times = []
    for step in range(CHUNK_SIZE):
        action_times.append(step / meta.fps)

    dataset = LeRobotDataset(REPO_ID, root=DATA_ROOT, delta_timestamps={"action": action_times})

    focus_task_indices = task_indices_of_suite[FOCUS_SUITE]
    episodes_of_task = group_episodes_by_task(dataset, meta)
    print_episode_lengths(meta, focus_task_indices, episodes_of_task)

    first_episode = meta.episodes[0]
    print_sample(dataset[first_episode["dataset_from_index"]], "sample at the start of episode 0:")
    # Near the end of an episode there are no future actions left, is_pad marks that padding
    print_sample(dataset[first_episode["dataset_to_index"] - 1], "sample at the last frame of episode 0:")

    # the first episode of each of the first few focus tasks
    picture_episodes = []
    for task_index in focus_task_indices[:PICTURE_EPISODES]:
        picture_episodes.append(episodes_of_task[task_index][0])
    draw_picture(dataset, picture_episodes)


if __name__ == "__main__":
    main()
