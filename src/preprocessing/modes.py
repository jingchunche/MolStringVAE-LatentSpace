"""Convert canonical/enumerated tokens into training pairs for each mode."""


OUTPUT_KEYS = (
    "one_tokens",
    "input_tokens",
    "target_tokens",
)


def empty_outputs():
    return {key: [] for key in OUTPUT_KEYS}


def required_variants(mode, configured_num_variants):
    if mode == "can2can":
        return 0
    if mode == "multi-enum2can":
        return configured_num_variants
    return 1


def build_mode_outputs(mode, canonical, variants, num_variants):
    """Build training pairs and one representative input per molecule."""
    outputs = empty_outputs()

    if mode == "can2can":
        outputs["input_tokens"].append(canonical)
        outputs["target_tokens"].append(canonical)
        fallback_count = 0
    elif mode == "enum2can":
        selected = list(variants[:1]) or [canonical]
        fallback_count = int(not variants)
        outputs["input_tokens"].append(selected[0])
        outputs["target_tokens"].append(canonical)
    elif mode == "multi-enum2can":
        selected = list(variants[:num_variants])
        fallback_count = max(0, num_variants - len(selected))
        training_inputs = selected or [canonical]
        outputs["input_tokens"].extend(training_inputs)
        outputs["target_tokens"].extend([canonical] * len(training_inputs))
    elif mode == "enum2can+can2enum":
        selected = list(variants[:1])
        fallback_count = int(not selected)
        if selected:
            outputs["input_tokens"].extend([selected[0], canonical])
            outputs["target_tokens"].extend([canonical, selected[0]])
        else:
            outputs["input_tokens"].append(canonical)
            outputs["target_tokens"].append(canonical)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    # Keep exactly one molecule-level entry, selected from this molecule's
    # training inputs even when the selected mode produces multiple pairs.
    outputs["one_tokens"].append(outputs["input_tokens"][0])
    return outputs, fallback_count
