from typing import Generator

import torch.nn as nn


def get_layer_candidates(module: nn.Module, max_depth: int = 1) -> Generator[str, None, None]:
    """Get a list of available layer candidates of the nn.Module in dot notation.

    Args:
        max_depth: The max-depth of layers traversed. Defaults to 1.

    Returns:
        Generator: A generator of the model's layer candidates in dot notation.
    """

    assert max_depth >= 0, 'max_depth must be a non-negative integer'
    assert isinstance(module, nn.Module), 'model must be a torch.nn.Module'

    def get_modules(model: nn.Module, prefix: str = '', depth: int = 0) -> Generator[str, None, None]:
        if prefix:
            yield prefix

        for name, m in model.named_children():
            if depth >= max_depth:
                continue
            submodule_prefix = prefix + ('.' if prefix else '') + name
            yield from get_modules(m, submodule_prefix, depth + 1)

    return get_modules(module)
