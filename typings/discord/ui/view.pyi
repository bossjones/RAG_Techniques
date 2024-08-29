"""
This type stub file was generated by pyright.
"""

from typing import Any, ClassVar, Coroutine, Dict, List, Optional, Sequence, TYPE_CHECKING
from .item import Item, ItemCallbackType
from typing_extensions import Self
from ..interactions import Interaction
from ..message import Message
from ..types.components import Component as ComponentPayload
from ..types.interactions import ModalSubmitComponentInteractionData as ModalSubmitComponentInteractionDataPayload
from ..state import ConnectionState

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
__all__ = ('View', )
if TYPE_CHECKING:
    ...
_log = ...
class _ViewWeights:
    __slots__ = ...
    def __init__(self, children: List[Item]) -> None:
        ...

    def find_open_space(self, item: Item) -> int:
        ...

    def add_item(self, item: Item) -> None:
        ...

    def remove_item(self, item: Item) -> None:
        ...

    def clear(self) -> None:
        ...



class _ViewCallback:
    __slots__ = ...
    def __init__(self, callback: ItemCallbackType[Any, Any], view: View, item: Item[View]) -> None:
        ...

    def __call__(self, interaction: Interaction) -> Coroutine[Any, Any, Any]:
        ...



class View:
    """Represents a UI view.

    This object must be inherited to create a UI within Discord.

    .. versionadded:: 2.0

    Parameters
    -----------
    timeout: Optional[:class:`float`]
        Timeout in seconds from last interaction with the UI before no longer accepting input.
        If ``None`` then there is no timeout.
    """
    __discord_ui_view__: ClassVar[bool] = ...
    __discord_ui_modal__: ClassVar[bool] = ...
    __view_children_items__: ClassVar[List[ItemCallbackType[Any, Any]]] = ...
    def __init_subclass__(cls) -> None:
        ...

    def __init__(self, *, timeout: Optional[float] = ...) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def to_components(self) -> List[Dict[str, Any]]:
        ...

    @property
    def timeout(self) -> Optional[float]:
        """Optional[:class:`float`]: The timeout in seconds from last interaction with the UI before no longer accepting input.
        If ``None`` then there is no timeout.
        """
        ...

    @timeout.setter
    def timeout(self, value: Optional[float]) -> None:
        ...

    @property
    def children(self) -> List[Item[Self]]:
        """List[:class:`Item`]: The list of children attached to this view."""
        ...

    @classmethod
    def from_message(cls, message: Message, /, *, timeout: Optional[float] = ...) -> View:
        """Converts a message's components into a :class:`View`.

        The :attr:`.Message.components` of a message are read-only
        and separate types from those in the ``discord.ui`` namespace.
        In order to modify and edit message components they must be
        converted into a :class:`View` first.

        Parameters
        -----------
        message: :class:`discord.Message`
            The message with components to convert into a view.
        timeout: Optional[:class:`float`]
            The timeout of the converted view.

        Returns
        --------
        :class:`View`
            The converted view. This always returns a :class:`View` and not
            one of its subclasses.
        """
        ...

    def add_item(self, item: Item[Any]) -> Self:
        """Adds an item to the view.

        This function returns the class instance to allow for fluent-style
        chaining.

        Parameters
        -----------
        item: :class:`Item`
            The item to add to the view.

        Raises
        --------
        TypeError
            An :class:`Item` was not passed.
        ValueError
            Maximum number of children has been exceeded (25)
            or the row the item is trying to be added to is full.
        """
        ...

    def remove_item(self, item: Item[Any]) -> Self:
        """Removes an item from the view.

        This function returns the class instance to allow for fluent-style
        chaining.

        Parameters
        -----------
        item: :class:`Item`
            The item to remove from the view.
        """
        ...

    def clear_items(self) -> Self:
        """Removes all items from the view.

        This function returns the class instance to allow for fluent-style
        chaining.
        """
        ...

    async def interaction_check(self, interaction: Interaction, /) -> bool:
        """|coro|

        A callback that is called when an interaction happens within the view
        that checks whether the view should process item callbacks for the interaction.

        This is useful to override if, for example, you want to ensure that the
        interaction author is a given user.

        The default implementation of this returns ``True``.

        .. note::

            If an exception occurs within the body then the check
            is considered a failure and :meth:`on_error` is called.

        Parameters
        -----------
        interaction: :class:`~discord.Interaction`
            The interaction that occurred.

        Returns
        ---------
        :class:`bool`
            Whether the view children's callbacks should be called.
        """
        ...

    async def on_timeout(self) -> None:
        """|coro|

        A callback that is called when a view's timeout elapses without being explicitly stopped.
        """
        ...

    async def on_error(self, interaction: Interaction, error: Exception, item: Item[Any], /) -> None:
        """|coro|

        A callback that is called when an item's callback or :meth:`interaction_check`
        fails with an error.

        The default implementation logs to the library logger.

        Parameters
        -----------
        interaction: :class:`~discord.Interaction`
            The interaction that led to the failure.
        error: :class:`Exception`
            The exception that was raised.
        item: :class:`Item`
            The item that failed the dispatch.
        """
        ...

    def stop(self) -> None:
        """Stops listening to interaction events from this view.

        This operation cannot be undone.
        """
        ...

    def is_finished(self) -> bool:
        """:class:`bool`: Whether the view has finished interacting."""
        ...

    def is_dispatching(self) -> bool:
        """:class:`bool`: Whether the view has been added for dispatching purposes."""
        ...

    def is_persistent(self) -> bool:
        """:class:`bool`: Whether the view is set up as persistent.

        A persistent view has all their components with a set ``custom_id`` and
        a :attr:`timeout` set to ``None``.
        """
        ...

    async def wait(self) -> bool:
        """|coro|

        Waits until the view has finished interacting.

        A view is considered finished when :meth:`stop` is called
        or it times out.

        Returns
        --------
        :class:`bool`
            If ``True``, then the view timed out. If ``False`` then
            the view finished normally.
        """
        ...



class ViewStore:
    def __init__(self, state: ConnectionState) -> None:
        ...

    @property
    def persistent_views(self) -> Sequence[View]:
        ...

    def add_view(self, view: View, message_id: Optional[int] = ...) -> None:
        ...

    def remove_view(self, view: View) -> None:
        ...

    def dispatch_view(self, component_type: int, custom_id: str, interaction: Interaction) -> None:
        ...

    def dispatch_modal(self, custom_id: str, interaction: Interaction, components: List[ModalSubmitComponentInteractionDataPayload]) -> None:
        ...

    def remove_interaction_mapping(self, interaction_id: int) -> None:
        ...

    def is_message_tracked(self, message_id: int) -> bool:
        ...

    def remove_message_tracking(self, message_id: int) -> Optional[View]:
        ...

    def update_from_message(self, message_id: int, data: List[ComponentPayload]) -> None:
        ...
