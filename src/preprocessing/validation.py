"""Configuration validation for the unified preprocessing pipeline."""

REPRESENTATIONS = {"smiles", "selfies", "group-selfies"}
MODES = {"can2can", "enum2can", "multi-enum2can", "enum2can+can2enum"}
GROUP_STRATEGIES = {"none", "masking", "traversal", "masking+traversal"}


def validate_config(config):
    representation = config["representation"]
    mode = config["mode"]
    strategy = config["group_selfies"]["strategy"]
    num_variants = config["enumeration"]["num_variants"]
    max_attempts = config["enumeration"]["max_attempts"]

    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unsupported representation: {representation}")
    if mode not in MODES:
        raise ValueError(f"Unsupported mode: {mode}")
    if strategy not in GROUP_STRATEGIES:
        raise ValueError(f"Unsupported Group SELFIES strategy: {strategy}")
    if representation != "group-selfies" and strategy != "none":
        raise ValueError(
            "group_selfies.strategy must be 'none' unless representation is "
            "'group-selfies'"
        )
    if representation == "group-selfies" and strategy == "none" and mode != "can2can":
        raise ValueError(
            "Group SELFIES enumeration modes require masking, traversal, or "
            "masking+traversal"
        )
    if num_variants < 1:
        raise ValueError("enumeration.num_variants must be at least 1")
    if mode == "multi-enum2can" and num_variants < 2:
        raise ValueError("multi-enum2can requires enumeration.num_variants >= 2")
    if max_attempts < 1:
        raise ValueError("enumeration.max_attempts must be at least 1")
    if config["runtime"]["max_workers"] < 1:
        raise ValueError("runtime.max_workers must be at least 1")
    if config["runtime"]["chunk_size"] < 1:
        raise ValueError("runtime.chunk_size must be at least 1")

