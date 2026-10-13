"""The LIBERO demonstrations, turned into training samples.

Reads the dataset that data/libero_data.py downloaded, and returns one sample per frame:

    images        (num_cameras, 3, S, S) float in [0, 1]
    state         (8,) raw robot state
    action        (chunk, 7) raw actions; the tail past the episode end is padding
    action_is_pad (chunk,) bool, True on that padding
    text_ids      (T,) SmolVLM2 token ids of the instruction
    text_mask     (T,) True on real tokens, False on the padding
    task          the instruction string, used for the language ablation

Normalisation is deliberately NOT done here. The policy owns it, so the statistics travel
inside the checkpoint and act() can undo them; see models/policies/.

The dataset numbers its 40 tasks in its own order, so a LIBERO task id is not a dataset
task index. Everything here matches instructions by their text instead of hard-coding
that mapping.
"""

import numpy as np
import torch
from libero.libero import benchmark
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from transformers import AutoProcessor

REPO_ID = "lerobot/libero"

# the frozen trunk; its tokenizer is the one the language tokens have to fit
TRUNK_NAME = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

# both cameras, in the order the model receives them
CAMERA_KEYS = ["observation.images.image", "observation.images.image2"]


def load_tokenizer():
    """The tokenizer of the frozen trunk.

    Not mini-vla's word-level tokenizer: these ids are looked up in SmolVLM2's embedding
    table, so they have to be SmolVLM2's own ids.
    """
    processor = AutoProcessor.from_pretrained(TRUNK_NAME)
    return processor.tokenizer


def suite_languages(suite_name):
    """The instruction strings of the tasks in one LIBERO suite, in LIBERO task order."""
#     suite_languages("libero_goal")
# → ['open the middle drawer of the cabinet', 'put the bowl on the stove', ...,
#    'put the wine bottle on the rack']
    suite = benchmark.get_benchmark_dict()[suite_name]()
    languages = []
    for task_id in range(suite.n_tasks):
        languages.append(suite.get_task(task_id).language)
    return languages


def dataset_task_indices(meta, languages):
    """Find the dataset's task number for each instruction string.

    A language that is not in the dataset raises, rather than quietly training on fewer
    tasks than were asked for.
    """
    index_of_language = meta.tasks["task_index"].to_dict()
    missing = []
    for language in languages:
        if language not in index_of_language:
            missing.append(language)
    if missing:
        raise KeyError("these instructions are not in the dataset: " + str(missing))

    task_indices = []
    for language in languages:
        task_indices.append(int(index_of_language[language]))
    return task_indices


def episodes_for_tasks(dataset, task_indices, overfit_task_index):
    """The episodes belonging to the given dataset task numbers.

    An episode's task is not stored in the episode table, so read the task of its first
    frame. `overfit_task_index` keeps a single task when it is not None, which is how a
    run is narrowed down to one task for the overfit check.
    """
    task_index_column = dataset.hf_dataset.data.column("task_index").to_numpy()
    language_of_index = {}
    for language, index in dataset.meta.tasks["task_index"].to_dict().items():
        language_of_index[int(index)] = language

    episodes = []
    for episode in dataset.meta.episodes:
        first_frame = int(episode["dataset_from_index"])
        task_index = int(task_index_column[first_frame])
        if task_index not in task_indices:
            continue
        if overfit_task_index is not None and task_index != overfit_task_index:
            continue
        one_episode = {
            "first_frame": first_frame,
            "length": int(episode["length"]),
            "task": language_of_index[task_index],
        }
        episodes.append(one_episode)
    return episodes


def split_episodes(episodes, val_fraction, seed):
    """Split the episodes, not the frames, into a training and a validation set.

    Splitting by frame would put two frames of the same demonstration on both sides, and
    the validation loss would look better than the model really is.
    """
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(episodes), generator=generator).tolist()
    num_val = max(1, round(len(episodes) * val_fraction))

    val_episodes = []
    for position in order[:num_val]:
        val_episodes.append(episodes[position])
    train_episodes = []
    for position in order[num_val:]:
        train_episodes.append(episodes[position])
    return train_episodes, val_episodes


def frame_indices_of(episodes):
    """Every frame index of the given episodes, in order."""
    indices = []
    for episode in episodes:
        for offset in range(episode["length"]):
            indices.append(episode["first_frame"] + offset)
    return indices


def read_state_and_action_columns(dataset):
    """The whole dataset's state and action, as two (N, D) float arrays.

    Reads only these two columns, so no video is decoded.
    """
    columns = dataset.hf_dataset.select_columns(["observation.state", "action"])[:]
    states = np.asarray(columns["observation.state"], dtype=np.float32)
    actions = np.asarray(columns["action"], dtype=np.float32)
    return states, actions


def compute_norm_stats(states, actions, frame_indices):
    """Mean and standard deviation of state and action over the given frames.

    Computed from the training frames only - using the validation frames too would leak a
    little of the validation set into training. The floor on the standard deviation keeps
    a constant dimension from dividing by zero.
    """
    train_states = states[frame_indices].astype(np.float64)
    train_actions = actions[frame_indices].astype(np.float64)
    stats = {
        "state_mean": torch.tensor(train_states.mean(0), dtype=torch.float32),
        "state_std": torch.tensor(np.maximum(train_states.std(0), 1e-6), dtype=torch.float32),
        "action_mean": torch.tensor(train_actions.mean(0), dtype=torch.float32),
        "action_std": torch.tensor(np.maximum(train_actions.std(0), 1e-6), dtype=torch.float32),
    }
    return stats


def tokenize_instructions(tokenizer, languages):
    """Instruction string -> (ids, mask), every one padded to the same length.

    All samples in a batch need the same text length, so the whole set is padded to the
    longest instruction. SmolVLM2 does not need a beginning-of-sequence token here: the
    trunk receives these as raw embedding lookups, not as a chat prompt.
    """
    id_lists = []
    for language in languages:
        id_lists.append(tokenizer(language)["input_ids"])
    longest = max(len(ids) for ids in id_lists)

    instructions = {}
    for language, ids in zip(languages, id_lists):
        padding = longest - len(ids)
        mask = [True] * len(ids) + [False] * padding
        padded_ids = ids + [tokenizer.pad_token_id] * padding
        instructions[language] = (torch.tensor(padded_ids, dtype=torch.long),
                                  torch.tensor(mask, dtype=torch.bool))
    return instructions


class LiberoSamples(torch.utils.data.Dataset):
    """One frame -> one training sample.

    `frame_indices` selects which frames this instance covers, so the training and the
    validation set are two instances over the same underlying dataset.
    """

    def __init__(self, dataset, frame_indices, instructions, camera_size):
        self.dataset = dataset
        self.frame_indices = frame_indices
        self.instructions = instructions
        self.camera_size = camera_size

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, index):
        frame = self.dataset[self.frame_indices[index]]
        text_ids, text_mask = self.instructions[frame["task"]]
        sample = {
            "images": self.stack_cameras(frame),
            "state": frame["observation.state"],
            "action": frame["action"],
            "action_is_pad": frame["action_is_pad"],
            "text_ids": text_ids,
            "text_mask": text_mask,
            "task": frame["task"],
        }
        return sample

    def stack_cameras(self, frame):
        """Both camera images as (num_cameras, 3, S, S), resized to the model's size."""
        camera_images = []
        for key in CAMERA_KEYS:
            image = frame[key].float()
            if image.shape[-1] != self.camera_size:
                # bilinear rather than nearest: the images are photographs, and nearest
                # would turn a 2x downscale into an alias-heavy pixel pick
                image = torch.nn.functional.interpolate(
                    image.unsqueeze(0),
                    size=(self.camera_size, self.camera_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            camera_images.append(image)
        return torch.stack(camera_images)


def build(config):
    """Build both sample sets and the normalisation statistics from a config.

    Returns (train_samples, val_samples, norm_stats). `norm_stats` is computed from the
    training frames only and is meant to be stored inside the checkpoint.
    """
    data = config["data"]

    # the metadata is read first because the chunk timestamps are in seconds and need fps
    meta = LeRobotDatasetMetadata(REPO_ID, root=data["dataset_root"])
    dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=data["dataset_root"],
        delta_timestamps=action_timestamps(data["chunk_size"], meta.fps),
    )

    languages = suite_languages(data["suite"])
    task_indices = dataset_task_indices(dataset.meta, languages)

    overfit_task_index = None
    if data["overfit_task"] is not None:
        overfit_task_index = task_indices[data["overfit_task"]]# overfit是用来自检的

    episodes = episodes_for_tasks(dataset, task_indices, overfit_task_index)
    train_episodes, val_episodes = split_episodes(episodes, data["val_episode_frac"],
                                                  seed=0)

    train_frames = frame_indices_of(train_episodes)
    val_frames = frame_indices_of(val_episodes)

    states, actions = read_state_and_action_columns(dataset)
    norm_stats = compute_norm_stats(states, actions, train_frames)

    # only the instructions actually used, so the padding length is as short as possible
    used_languages = []
    for episode in episodes:
        if episode["task"] not in used_languages:
            used_languages.append(episode["task"])
    instructions = tokenize_instructions(load_tokenizer(), used_languages)

    train_samples = LiberoSamples(dataset, train_frames, instructions, data["camera_size"])
    val_samples = LiberoSamples(dataset, val_frames, instructions, data["camera_size"])
    return train_samples, val_samples, norm_stats


def action_timestamps(chunk_size, fps):
    """The delta_timestamps that make the dataset return a chunk of future actions.

    Times are in seconds, so step i is i / fps seconds after the current frame. The dataset
    returns `action` of shape (chunk_size, action_dim) plus `action_is_pad`, and it never
    reads past the end of an episode: the tail is filled and marked instead.
    """
    timestamps = []
    for step in range(chunk_size):
        timestamps.append(step / fps)
    return {"action": timestamps}
