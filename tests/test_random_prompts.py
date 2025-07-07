import pytest

from mesh.subnet.utils.random_prompts import RandomPrompts

# pytest tests/test_random_prompts.py -rP


# pytest tests/test_random_prompts.py::test_random_prompts_init -rP

def test_random_prompts_init():
    random_prompts = RandomPrompts("bigscience/bloom-560m")
    assert random_prompts.tokenizer is not None
    assert len(random_prompts.templates) > 0
    assert len(random_prompts.fillers) > 0

# pytest tests/test_random_prompts.py::test_generate_prompt_tensor -rP

def test_generate_prompt_tensor():
    random_prompts = RandomPrompts("bigscience/bloom-560m")
    random_prompt = random_prompts.generate_prompt_tensor()
    assert random_prompt is not None
    assert len(random_prompt) > 0
