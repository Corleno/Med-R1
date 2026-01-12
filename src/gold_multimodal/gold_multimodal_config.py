from dataclasses import dataclass, field
from typing import Optional
from trl import ScriptArguments

from ..gold.gold_config import GOLDConfig


@dataclass
class GOLDMultimodalScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: [],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


@dataclass
class GOLDMultimodalConfig(GOLDConfig):
    r"""
    Configuration class for [`GOLDMultimodalTrainer`].
    """

    # "vqa" or "vqa_thinking"
    dataset_type: str = "vqa"
    dataset_from_disk: bool = True

    # With alpha = 1.0, the model will perform only distillation.
    # With alpha = 0.0, the model will perform GPRO only.
    alpha: float = 1.0

    # Tau refers to the threshold for the on-policy knowledge distillation. 
    # If the probability of the on-policy knowledge distillation is greater than tau, 
    # the on-policy distillation will be performed. Otherwise, the off-policy distillation will be performed.
    tau: float = None

    # KL coefficient. If `0.0` (default), the reference model is not loaded, reducing memory usage and
    # improving training speed. [DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement
    # learning](https://huggingface.co/papers/2501.12948) use a value of `0.001`.
    beta_rl: float = 0.0 

    # Number of generations to sample for GPRO.
    num_generations: int = 8

    # Number of knowledge enhancement samples to generate
    num_knowledge_enhancement: int = 0