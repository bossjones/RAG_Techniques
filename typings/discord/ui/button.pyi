"""
This type stub file was generated by pyright.
"""

from typing import Callable, Literal, Optional, TYPE_CHECKING, Tuple, TypeVar, Union
from .item import Item, ItemCallbackType
from ..enums import ButtonStyle, ComponentType
from ..partial_emoji import PartialEmoji
from ..components import Button as ButtonComponent
from typing_extensions import Self
from ..emoji import Emoji
from ..types.components import ButtonComponent as ButtonComponentPayload

"""
The MIT License (MIT)

Copyright (c) 2015-present Rapptz

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
__all__ = ('Button', 'button')
if TYPE_CHECKING:
    ...
V = TypeVar('V', bound='View', covariant=True)
class Button(Item[V]):
    """Represents a UI button.

    .. versionadded:: 2.0

    Parameters
    ------------
    style: :class:`discord.ButtonStyle`
        The style of the button.
    custom_id: Optional[:class:`str`]
        The ID of the button that gets received during an interaction.
        If this button is for a URL, it does not have a custom ID.
    url: Optional[:class:`str`]
        The URL this button sends you to.
    disabled: :class:`bool`
        Whether the button is disabled or not.
    label: Optional[:class:`str`]
        The label of the button, if any.
    emoji: Optional[Union[:class:`.PartialEmoji`, :class:`.Emoji`, :class:`str`]]
        The emoji of the button, if available.
    row: Optional[:class:`int`]
        The relative row this button belongs to. A Discord component can only have 5
        rows. By default, items are arranged automatically into those 5 rows. If you'd
        like to control the relative positioning of the row then passing an index is advised.
        For example, row=1 will show up before row=2. Defaults to ``None``, which is automatic
        ordering. The row number must be between 0 and 4 (i.e. zero indexed).
    """
    __item_repr_attributes__: Tuple[str, ...] = ...
    def __init__(self, *, style: ButtonStyle = ..., label: Optional[str] = ..., disabled: bool = ..., custom_id: Optional[str] = ..., url: Optional[str] = ..., emoji: Optional[Union[str, Emoji, PartialEmoji]] = ..., row: Optional[int] = ...) -> None:
        ...

    @property
    def style(self) -> ButtonStyle:
        """:class:`discord.ButtonStyle`: The style of the button."""
        ...

    @style.setter
    def style(self, value: ButtonStyle) -> None:
        ...

    @property
    def custom_id(self) -> Optional[str]:
        """Optional[:class:`str`]: The ID of the button that gets received during an interaction.

        If this button is for a URL, it does not have a custom ID.
        """
        ...

    @custom_id.setter
    def custom_id(self, value: Optional[str]) -> None:
        ...

    @property
    def url(self) -> Optional[str]:
        """Optional[:class:`str`]: The URL this button sends you to."""
        ...

    @url.setter
    def url(self, value: Optional[str]) -> None:
        ...

    @property
    def disabled(self) -> bool:
        """:class:`bool`: Whether the button is disabled or not."""
        ...

    @disabled.setter
    def disabled(self, value: bool) -> None:
        ...

    @property
    def label(self) -> Optional[str]:
        """Optional[:class:`str`]: The label of the button, if available."""
        ...

    @label.setter
    def label(self, value: Optional[str]) -> None:
        ...

    @property
    def emoji(self) -> Optional[PartialEmoji]:
        """Optional[:class:`.PartialEmoji`]: The emoji of the button, if available."""
        ...

    @emoji.setter
    def emoji(self, value: Optional[Union[str, Emoji, PartialEmoji]]) -> None:
        ...

    @classmethod
    def from_component(cls, button: ButtonComponent) -> Self:
        ...

    @property
    def type(self) -> Literal[ComponentType.button]:
        ...

    def to_component_dict(self) -> ButtonComponentPayload:
        ...

    def is_dispatchable(self) -> bool:
        ...

    def is_persistent(self) -> bool:
        ...



def button(*, label: Optional[str] = ..., custom_id: Optional[str] = ..., disabled: bool = ..., style: ButtonStyle = ..., emoji: Optional[Union[str, Emoji, PartialEmoji]] = ..., row: Optional[int] = ...) -> Callable[[ItemCallbackType[V, Button[V]]], Button[V]]:
    """A decorator that attaches a button to a component.

    The function being decorated should have three parameters, ``self`` representing
    the :class:`discord.ui.View`, the :class:`discord.Interaction` you receive and
    the :class:`discord.ui.Button` being pressed.

    .. note::

        Buttons with a URL cannot be created with this function.
        Consider creating a :class:`Button` manually instead.
        This is because buttons with a URL do not have a callback
        associated with them since Discord does not do any processing
        with it.

    Parameters
    ------------
    label: Optional[:class:`str`]
        The label of the button, if any.
    custom_id: Optional[:class:`str`]
        The ID of the button that gets received during an interaction.
        It is recommended not to set this parameter to prevent conflicts.
    style: :class:`.ButtonStyle`
        The style of the button. Defaults to :attr:`.ButtonStyle.grey`.
    disabled: :class:`bool`
        Whether the button is disabled or not. Defaults to ``False``.
    emoji: Optional[Union[:class:`str`, :class:`.Emoji`, :class:`.PartialEmoji`]]
        The emoji of the button. This can be in string form or a :class:`.PartialEmoji`
        or a full :class:`.Emoji`.
    row: Optional[:class:`int`]
        The relative row this button belongs to. A Discord component can only have 5
        rows. By default, items are arranged automatically into those 5 rows. If you'd
        like to control the relative positioning of the row then passing an index is advised.
        For example, row=1 will show up before row=2. Defaults to ``None``, which is automatic
        ordering. The row number must be between 0 and 4 (i.e. zero indexed).
    """
    ...
