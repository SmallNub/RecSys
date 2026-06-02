from typing import List

import fsspec
import omegaconf
from fsspec.implementations.arrow import HadoopFileSystem


def patch_fsspec():
    if "viewfs" not in fsspec.registry or "viewfs" not in fsspec.available_protocols():
        fsspec.register_implementation("viewfs", HadoopFileSystem)


def _resolve(cfg, key, to_resolve):
    value = cfg._get_node(key)._value()
    if isinstance(value, str) and any(f"${{{c}:" in value for c in to_resolve):
        cfg[key] = cfg[key]  # Resolved value


def selective_oc_resolver(cfg, which: List[str] = []):
    # cfg : OmegaConf object
    # to_resolve : list of resolvers to resolve (ex : env, now, ...)

    if isinstance(cfg, omegaconf.DictConfig):
        iterator = cfg.items
    elif isinstance(cfg, omegaconf.ListConfig):
        iterator = lambda: enumerate(cfg)  # noqa: E731

    for k, v in iterator():
        if v.__class__ in [omegaconf.DictConfig, omegaconf.ListConfig]:
            selective_oc_resolver(v, which)
        else:
            _resolve(cfg, k, which)
