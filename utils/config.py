"""Read an experiment YAML, then let the command line override a few scalars on top.

The YAML file is the experiment. The command line exists only to override values that
are already in the YAML (learning rate, batch size, and so on) so a quick run does not
need a new file. It cannot introduce a setting the YAML does not have: naming a missing
key raises, so a typo fails loudly instead of silently training something else.
"""

import copy

import yaml


def load(path):
    """Parse `path` and return the config as a plain dict."""
    with open(path) as file:
        config = yaml.safe_load(file)
    return config


def with_overrides(config, overrides):
    """Return a copy of `config` with the given dotted paths replaced.

    overrides: {"train.lr": 1e-5, "data.camera_size": 512}. A value of None means "this
    flag was not given" and is ignored, so scripts can pass their whole argument dict.
    """
    merged = copy.deepcopy(config)
    for dotted_path in overrides:
        value = overrides[dotted_path]
        if value is None:
            continue

        keys = dotted_path.split(".")
        section = merged
        for key in keys[:-1]:
            if key not in section:
                raise KeyError(f"config has no section '{key}', "
                               f"named by the override '{dotted_path}'")
            section = section[key]

        last_key = keys[-1]
        if last_key not in section:
            raise KeyError(f"config has no key '{last_key}' in section "
                           f"'{'.'.join(keys[:-1])}', named by the override '{dotted_path}'")
        section[last_key] = value

    return merged
