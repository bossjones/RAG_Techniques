"""
This type stub file was generated by pyright.
"""

from typing import Callable, List, Optional, Union
from tf_keras.optimizers.legacy import Adam
from .modeling_tf_utils import keras

"""Functions and classes related to optimization (weight updates)."""
if hasattr(keras.optimizers.schedules, "learning_rate_schedule"):
    schedules = ...
else:
    schedules = ...
class WarmUp(schedules.LearningRateSchedule):
    """
    Applies a warmup schedule on a given learning rate decay schedule.

    Args:
        initial_learning_rate (`float`):
            The initial learning rate for the schedule after the warmup (so this will be the learning rate at the end
            of the warmup).
        decay_schedule_fn (`Callable`):
            The schedule function to apply after the warmup for the rest of training.
        warmup_steps (`int`):
            The number of steps for the warmup part of training.
        power (`float`, *optional*, defaults to 1.0):
            The power to use for the polynomial warmup (defaults is a linear warmup).
        name (`str`, *optional*):
            Optional name prefix for the returned tensors during the schedule.
    """
    def __init__(self, initial_learning_rate: float, decay_schedule_fn: Callable, warmup_steps: int, power: float = ..., name: str = ...) -> None:
        ...

    def __call__(self, step):
        ...

    def get_config(self): # -> dict[str, Any]:
        ...



def create_optimizer(init_lr: float, num_train_steps: int, num_warmup_steps: int, min_lr_ratio: float = ..., adam_beta1: float = ..., adam_beta2: float = ..., adam_epsilon: float = ..., adam_clipnorm: Optional[float] = ..., adam_global_clipnorm: Optional[float] = ..., weight_decay_rate: float = ..., power: float = ..., include_in_weight_decay: Optional[List[str]] = ...): # -> tuple[AdamWeightDecay | Any, WarmUp | Any]:
    """
    Creates an optimizer with a learning rate schedule using a warmup phase followed by a linear decay.

    Args:
        init_lr (`float`):
            The desired learning rate at the end of the warmup phase.
        num_train_steps (`int`):
            The total number of training steps.
        num_warmup_steps (`int`):
            The number of warmup steps.
        min_lr_ratio (`float`, *optional*, defaults to 0):
            The final learning rate at the end of the linear decay will be `init_lr * min_lr_ratio`.
        adam_beta1 (`float`, *optional*, defaults to 0.9):
            The beta1 to use in Adam.
        adam_beta2 (`float`, *optional*, defaults to 0.999):
            The beta2 to use in Adam.
        adam_epsilon (`float`, *optional*, defaults to 1e-8):
            The epsilon to use in Adam.
        adam_clipnorm (`float`, *optional*, defaults to `None`):
            If not `None`, clip the gradient norm for each weight tensor to this value.
        adam_global_clipnorm (`float`, *optional*, defaults to `None`)
            If not `None`, clip gradient norm to this value. When using this argument, the norm is computed over all
            weight tensors, as if they were concatenated into a single vector.
        weight_decay_rate (`float`, *optional*, defaults to 0):
            The weight decay to use.
        power (`float`, *optional*, defaults to 1.0):
            The power to use for PolynomialDecay.
        include_in_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters except bias and layer norm parameters.
    """
    ...

class AdamWeightDecay(Adam):
    """
    Adam enables L2 weight decay and clip_by_global_norm on gradients. Just adding the square of the weights to the
    loss function is *not* the correct way of using L2 regularization/weight decay with Adam, since that will interact
    with the m and v parameters in strange ways as shown in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Instead we want to decay the weights in a manner that doesn't interact with the m/v parameters. This is equivalent
    to adding the square of the weights to the loss with plain (non-momentum) SGD.

    Args:
        learning_rate (`Union[float, LearningRateSchedule]`, *optional*, defaults to 0.001):
            The learning rate to use or a schedule.
        beta_1 (`float`, *optional*, defaults to 0.9):
            The beta1 parameter in Adam, which is the exponential decay rate for the 1st momentum estimates.
        beta_2 (`float`, *optional*, defaults to 0.999):
            The beta2 parameter in Adam, which is the exponential decay rate for the 2nd momentum estimates.
        epsilon (`float`, *optional*, defaults to 1e-07):
            The epsilon parameter in Adam, which is a small constant for numerical stability.
        amsgrad (`bool`, *optional*, defaults to `False`):
            Whether to apply AMSGrad variant of this algorithm or not, see [On the Convergence of Adam and
            Beyond](https://arxiv.org/abs/1904.09237).
        weight_decay_rate (`float`, *optional*, defaults to 0.0):
            The weight decay to apply.
        include_in_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to apply weight decay to. If none is passed, weight decay is
            applied to all parameters by default (unless they are in `exclude_from_weight_decay`).
        exclude_from_weight_decay (`List[str]`, *optional*):
            List of the parameter names (or re patterns) to exclude from applying weight decay to. If a
            `include_in_weight_decay` is passed, the names in it will supersede this list.
        name (`str`, *optional*, defaults to `"AdamWeightDecay"`):
            Optional name for the operations created when applying gradients.
        kwargs (`Dict[str, Any]`, *optional*):
            Keyword arguments. Allowed to be {`clipnorm`, `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
            norm; `clipvalue` is clip gradients by value, `decay` is included for backward compatibility to allow time
            inverse decay of learning rate. `lr` is included for backward compatibility, recommended to use
            `learning_rate` instead.
    """
    def __init__(self, learning_rate: Union[float, schedules.LearningRateSchedule] = ..., beta_1: float = ..., beta_2: float = ..., epsilon: float = ..., amsgrad: bool = ..., weight_decay_rate: float = ..., include_in_weight_decay: Optional[List[str]] = ..., exclude_from_weight_decay: Optional[List[str]] = ..., name: str = ..., **kwargs) -> None:
        ...

    @classmethod
    def from_config(cls, config):
        """Creates an optimizer from its config with WarmUp custom object."""
        ...

    def apply_gradients(self, grads_and_vars, name=..., **kwargs):
        ...

    def get_config(self):
        ...



class GradientAccumulator:
    """
    Gradient accumulation utility. When used with a distribution strategy, the accumulator should be called in a
    replica context. Gradients will be accumulated locally on each replica and without synchronization. Users should
    then call `.gradients`, scale the gradients if required, and pass the result to `apply_gradients`.
    """
    def __init__(self) -> None:
        """Initializes the accumulator."""
        ...

    @property
    def step(self):
        """Number of accumulated steps."""
        ...

    @property
    def gradients(self): # -> list[Any]:
        """The accumulated gradients on the current replica."""
        ...

    def __call__(self, gradients): # -> None:
        """Accumulates `gradients` on the current replica."""
        ...

    def reset(self): # -> None:
        """Resets the accumulated gradients on the current replica."""
        ...
