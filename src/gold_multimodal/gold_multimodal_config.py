from ..gold.gold_config import GOLDConfig

class GOLDMultimodalConfig(GOLDConfig):
    r"""
    Configuration class for [`GOLDMultimodalTrainer`].
    """
    dataset_type: str = "vqa" # "vqa" or "vqa_thinking"
    dataset_from_disk: bool = True