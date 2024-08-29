"""
This type stub file was generated by pyright.
"""

from typing import Any, List, Optional, TYPE_CHECKING
from .guild import Guild
from .types.template import Template as TemplatePayload
from .state import ConnectionState

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
__all__ = ('Template', )
if TYPE_CHECKING:
    ...
class _FriendlyHttpAttributeErrorHelper:
    __slots__ = ...
    def __getattr__(self, attr):
        ...



class _PartialTemplateState:
    def __init__(self, *, state) -> None:
        ...

    @property
    def shard_count(self):
        ...

    @property
    def user(self):
        ...

    @property
    def self_id(self):
        ...

    @property
    def member_cache_flags(self):
        ...

    def store_emoji(self, guild, packet) -> None:
        ...

    async def query_members(self, **kwargs: Any) -> List[Any]:
        ...

    def __getattr__(self, attr):
        ...



class Template:
    """Represents a Discord template.

    .. versionadded:: 1.4

    Attributes
    -----------
    code: :class:`str`
        The template code.
    uses: :class:`int`
        How many times the template has been used.
    name: :class:`str`
        The name of the template.
    description: :class:`str`
        The description of the template.
    creator: :class:`User`
        The creator of the template.
    created_at: :class:`datetime.datetime`
        An aware datetime in UTC representing when the template was created.
    updated_at: :class:`datetime.datetime`
        An aware datetime in UTC representing when the template was last updated.
        This is referred to as "last synced" in the official Discord client.
    source_guild: :class:`Guild`
        The guild snapshot that represents the data that this template currently holds.
    is_dirty: Optional[:class:`bool`]
        Whether the template has unsynced changes.

        .. versionadded:: 2.0
    """
    __slots__ = ...
    def __init__(self, *, state: ConnectionState, data: TemplatePayload) -> None:
        ...

    def __repr__(self) -> str:
        ...

    async def create_guild(self, name: str, icon: bytes = ...) -> Guild:
        """|coro|

        Creates a :class:`.Guild` using the template.

        Bot accounts in more than 10 guilds are not allowed to create guilds.

        .. versionchanged:: 2.0
            The ``region`` parameter has been removed.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ----------
        name: :class:`str`
            The name of the guild.
        icon: :class:`bytes`
            The :term:`py:bytes-like object` representing the icon. See :meth:`.ClientUser.edit`
            for more details on what is expected.

        Raises
        ------
        HTTPException
            Guild creation failed.
        ValueError
            Invalid icon image format given. Must be PNG or JPG.

        Returns
        -------
        :class:`.Guild`
            The guild created. This is not the same guild that is
            added to cache.
        """
        ...

    async def sync(self) -> Template:
        """|coro|

        Sync the template to the guild's current state.

        You must have :attr:`~Permissions.manage_guild` in the source guild to do this.

        .. versionadded:: 1.7

        .. versionchanged:: 2.0
            The template is no longer edited in-place, instead it is returned.

        Raises
        -------
        HTTPException
            Editing the template failed.
        Forbidden
            You don't have permissions to edit the template.
        NotFound
            This template does not exist.

        Returns
        --------
        :class:`Template`
            The newly edited template.
        """
        ...

    async def edit(self, *, name: str = ..., description: Optional[str] = ...) -> Template:
        """|coro|

        Edit the template metadata.

        You must have :attr:`~Permissions.manage_guild` in the source guild to do this.

        .. versionadded:: 1.7

        .. versionchanged:: 2.0
            The template is no longer edited in-place, instead it is returned.

        Parameters
        ------------
        name: :class:`str`
            The template's new name.
        description: Optional[:class:`str`]
            The template's new description.

        Raises
        -------
        HTTPException
            Editing the template failed.
        Forbidden
            You don't have permissions to edit the template.
        NotFound
            This template does not exist.

        Returns
        --------
        :class:`Template`
            The newly edited template.
        """
        ...

    async def delete(self) -> None:
        """|coro|

        Delete the template.

        You must have :attr:`~Permissions.manage_guild` in the source guild to do this.

        .. versionadded:: 1.7

        Raises
        -------
        HTTPException
            Editing the template failed.
        Forbidden
            You don't have permissions to edit the template.
        NotFound
            This template does not exist.
        """
        ...

    @property
    def url(self) -> str:
        """:class:`str`: The template url.

        .. versionadded:: 2.0
        """
        ...
