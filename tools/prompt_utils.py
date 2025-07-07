import random

def load_prompts(file_path: str) -> list[str]:
    """
    Load prompts from a text file, one per line.
    """
    with open(file_path, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

class PromptUtils:
    """
    Utilities for prompt engineering experiments.
    """

    @staticmethod
    def expand_prompts(base_prompts: list[str], templates: list[str]) -> list[str]:
        """
        Generate new prompt variants by formatting template strings with base prompts.

        Args:
            base_prompts: List of original prompt strings.
            templates: List of template strings containing '{prompt}' placeholder.

        Returns:
            List of expanded prompt strings.
        """
        expanded = []
        for prompt in base_prompts:
            for tpl in templates:
                expanded.append(tpl.format(prompt=prompt))
        return expanded

    @staticmethod
    def sample_prompts(prompts: list[str], k: int) -> list[str]:
        """
        Randomly sample k prompts from the list, with replacement minimized.
        """
        return random.sample(prompts, min(k, len(prompts)))
