from functools import reduce
import re

LORA_PATTERN = re.compile(r"\<lora:([^>]+):(-?[\d.]+)\>", re.DOTALL | re.MULTILINE)


def clean_lora(prompt: str):
    try:
        assert prompt
        return reduce(
            lambda r, m: r.replace(m.group(0)), LORA_PATTERN.finditer(prompt), prompt
        ).strip(", \n")
    except AssertionError:
        pass
    return prompt
