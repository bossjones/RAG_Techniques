"""
This type stub file was generated by pyright.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Callable, List, Optional, Union
from . import IntervalStrategy, PreTrainedTokenizerBase
from .modeling_tf_utils import keras

logger = ...
class KerasMetricCallback(keras.callbacks.Callback):
    """
    Callback to compute metrics at the end of every epoch. Unlike normal Keras metrics, these do not need to be
    compilable by TF. It is particularly useful for common NLP metrics like BLEU and ROUGE that require string
    operations or generation loops that cannot be compiled. Predictions (or generations) will be computed on the
    `eval_dataset` before being passed to the `metric_fn` in `np.ndarray` format. The `metric_fn` should compute
    metrics and return a dict mapping metric names to metric values.

    We provide an example of a suitable metric_fn that computes ROUGE scores for a summarization model below. Note that
    this example skips some post-processing for readability and simplicity, and should probably not be used as-is!

    ```py
    from datasets import load_metric

    rouge_metric = load_metric("rouge")


    def rouge_fn(predictions, labels):
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        result = rouge_metric.compute(predictions=decoded_predictions, references=decoded_labels)
        return {key: value.mid.fmeasure * 100 for key, value in result.items()}
    ```

    The above function will return a dict containing values which will be logged like any other Keras metric:

    ```
    {'rouge1': 37.4199, 'rouge2': 13.9768, 'rougeL': 34.361, 'rougeLsum': 35.0781
    ```

    Args:
        metric_fn (`Callable`):
            Metric function provided by the user. It will be called with two arguments - `predictions` and `labels`.
            These contain the model's outputs and matching labels from the dataset. It should return a dict mapping
            metric names to numerical values.
        eval_dataset (`tf.data.Dataset` or `dict` or `tuple` or `np.ndarray` or `tf.Tensor`):
            Validation data to be used to generate predictions for the `metric_fn`.
        output_cols (`List[str], *optional*):
            A list of columns to be retained from the model output as the predictions. Defaults to all.
        label_cols ('`List[str]`, *optional*'):
            A list of columns to be retained from the input dataset as the labels. Will be autodetected if this is not
            supplied.
        batch_size (`int`, *optional*):
            Batch size. Only used when the data is not a pre-batched `tf.data.Dataset`.
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            Whether we should use `model.generate()` to get outputs for the model.
        use_xla_generation (`bool`, *optional*, defaults to `False`):
            If we're generating, whether to compile model generation with XLA. This can massively increase the speed of
            generation (up to 100X speedup) but will require a new XLA compilation for each input shape. When using XLA
            generation, it's a good idea to pad your inputs to the same size, or to use the `pad_to_multiple_of`
            argument in your `tokenizer` or `DataCollator`, which will reduce the number of unique input shapes and
            save a lot of compilation time. This option has no effect is `predict_with_generate` is `False`.
        generate_kwargs (`dict`, *optional*):
            Keyword arguments to pass to `model.generate()` when generating. Has no effect if `predict_with_generate`
            is `False`.

    """
    def __init__(self, metric_fn: Callable, eval_dataset: Union[tf.data.Dataset, np.ndarray, tf.Tensor, tuple, dict], output_cols: Optional[List[str]] = ..., label_cols: Optional[List[str]] = ..., batch_size: Optional[int] = ..., predict_with_generate: bool = ..., use_xla_generation: bool = ..., generate_kwargs: Optional[dict] = ...) -> None:
        ...

    def on_epoch_end(self, epoch, logs=...): # -> None:
        ...



class PushToHubCallback(keras.callbacks.Callback):
    """
    Callback that will save and push the model to the Hub regularly. By default, it pushes once per epoch, but this can
    be changed with the `save_strategy` argument. Pushed models can be accessed like any other model on the hub, such
    as with the `from_pretrained` method.

    ```py
    from transformers.keras_callbacks import PushToHubCallback

    push_to_hub_callback = PushToHubCallback(
        output_dir="./model_save",
        tokenizer=tokenizer,
        hub_model_id="gpt5-7xlarge",
    )

    model.fit(train_dataset, callbacks=[push_to_hub_callback])
    ```

    Args:
        output_dir (`str`):
            The output directory where the model predictions and checkpoints will be written and synced with the
            repository on the Hub.
        save_strategy (`str` or [`~trainer_utils.IntervalStrategy`], *optional*, defaults to `"epoch"`):
            The checkpoint save strategy to adopt during training. Possible values are:

                - `"no"`: Save is done at the end of training.
                - `"epoch"`: Save is done at the end of each epoch.
                - `"steps"`: Save is done every `save_steps`
        save_steps (`int`, *optional*):
            The number of steps between saves when using the "steps" `save_strategy`.
        tokenizer (`PreTrainedTokenizerBase`, *optional*):
            The tokenizer used by the model. If supplied, will be uploaded to the repo alongside the weights.
        hub_model_id (`str`, *optional*):
            The name of the repository to keep in sync with the local `output_dir`. It can be a simple model ID in
            which case the model will be pushed in your namespace. Otherwise it should be the whole repository name,
            for instance `"user_name/model"`, which allows you to push to an organization you are a member of with
            `"organization_name/model"`.

            Will default to the name of `output_dir`.
        hub_token (`str`, *optional*):
            The token to use to push the model to the Hub. Will default to the token in the cache folder obtained with
            `huggingface-cli login`.
        checkpoint (`bool`, *optional*, defaults to `False`):
            Whether to save full training checkpoints (including epoch and optimizer state) to allow training to be
            resumed. Only usable when `save_strategy` is `"epoch"`.
    """
    def __init__(self, output_dir: Union[str, Path], save_strategy: Union[str, IntervalStrategy] = ..., save_steps: Optional[int] = ..., tokenizer: Optional[PreTrainedTokenizerBase] = ..., hub_model_id: Optional[str] = ..., hub_token: Optional[str] = ..., checkpoint: bool = ..., **model_card_args) -> None:
        ...

    def on_train_begin(self, logs=...): # -> None:
        ...

    def on_train_batch_end(self, batch, logs=...): # -> None:
        ...

    def on_epoch_end(self, epoch, logs=...): # -> None:
        ...

    def on_train_end(self, logs=...): # -> None:
        ...
