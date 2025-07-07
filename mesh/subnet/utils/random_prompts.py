import itertools
import random
from typing import List, Tuple

import torch
from transformers import AutoTokenizer

from mesh.subnet.utils.hoster import MAX_HOSTER_TOKENS

MIN_PROMPTS = 10

class RandomPrompts:
    def __init__(self, model_name_or_path: str):
        """
        As a node, you must create your own unique prompts for when you're the validator

        These prompts are used for hosters to generate an output for
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.templates = [
            "Describe the behavior of a {}.",
            "What happens when a {} is exposed to {}?",
            "Summarize the concept of {} in simple terms.",
            "How does {} compare to {}?",
            "Why is {} important in {}?",
            "Write a short story involving a {} and a {}.",
            "Explain how {} works using an analogy.",
            "List some pros and cons of {}.",
            "What are the ethical implications of {}?",
            "Predict the outcome of a {} involving {}."
        ]

        self.fillers = [
            "black hole", "quantum computer", "neural network", "ecosystem", "market crash", "pandemic",
            "AI agent", "time machine", "language model", "fusion reactor", "blockchain", "robot uprising"
        ]

        # Pre-validate templates
        self.valid_prompts = self._generate_valid_prompt_pool()

        if len(self.valid_prompts) < MIN_PROMPTS:
            raise ValueError(
                f"Only {len(self.valid_prompts)} valid prompts under {MIN_PROMPTS} tokens, "
                f"but at least {MIN_PROMPTS} are required."
            )

    def _generate_valid_prompt_pool(self) -> List[Tuple[str, List[str]]]:
        valid = []
        for template in self.templates:
            num_fields = template.count("{}")
            combinations = itertools.product(self.fillers, repeat=num_fields)
            for combo in combinations:
                prompt = template.format(*combo)
                token_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                if len(token_ids) <= MAX_HOSTER_TOKENS:
                    valid.append((template, list(combo)))
        return valid

    def generate_prompt_tensor(self) -> torch.Tensor:
        template = random.choice(self.templates)
        num_fields = template.count("{}")
        fills = random.sample(self.fillers, num_fields)
        prompt = template.format(*fills)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long)
