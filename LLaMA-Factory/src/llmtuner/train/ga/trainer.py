from types import MethodType
from typing import TYPE_CHECKING, Optional

from transformers import Trainer
from transformers.utils.versions import require_version

require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")

from ...extras.logging import get_logger
from ..utils import create_custom_optimzer, create_custom_scheduler


if TYPE_CHECKING:
    import torch

    from ...hparams import FinetuningArguments


logger = get_logger(__name__)


class CustomTrainer(Trainer):
    r"""
    Inherits Trainer for custom optimizer.
    """

    def __init__(self, finetuning_args: "FinetuningArguments", **kwargs) -> None:
        super().__init__(**kwargs)
        self.finetuning_args = finetuning_args
        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        r"""
        Computes the GA loss - maximizes rather than minimizes the loss to unlearn.

        Arguments:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                If True returns model outputs along with the loss.

        Returns:
            `Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]`:
                If return_outputs=False, returns the loss. Otherwise returns the loss and model outputs.
        """
        outputs = model(**inputs)
        loss = outputs.loss  # maximize rather than minimize the loss
        loss = -loss  # negate loss to maximize

        return (loss, outputs) if return_outputs else loss
