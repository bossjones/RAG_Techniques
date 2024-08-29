"""
This type stub file was generated by pyright.
"""

import discord.abc
from typing import List, Optional, TYPE_CHECKING, Union
from .asset import Asset
from .colour import Colour
from .flags import PublicUserFlags
from datetime import datetime
from .channel import DMChannel
from .guild import Guild
from .message import Message
from .state import ConnectionState
from .types.user import PartialUser as PartialUserPayload, User as UserPayload

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
if TYPE_CHECKING:
    ...
__all__ = ('User', 'ClientUser')
class _UserTag:
    __slots__ = ...
    id: int


class BaseUser(_UserTag):
    __slots__ = ...
    if TYPE_CHECKING:
        name: str
        id: int
        discriminator: str
        global_name: Optional[str]
        bot: bool
        system: bool
        _state: ConnectionState
        _avatar: Optional[str]
        _banner: Optional[str]
        _accent_colour: Optional[int]
        _public_flags: int
        ...
    def __init__(self, *, state: ConnectionState, data: Union[UserPayload, PartialUserPayload]) -> None:
        ...

    def __repr__(self) -> str:
        ...

    def __str__(self) -> str:
        ...

    def __eq__(self, other: object) -> bool:
        ...

    def __ne__(self, other: object) -> bool:
        ...

    def __hash__(self) -> int:
        ...

    @property
    def public_flags(self) -> PublicUserFlags:
        """:class:`PublicUserFlags`: The publicly available flags the user has."""
        ...

    @property
    def avatar(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns an :class:`Asset` for the avatar the user has.

        If the user has not uploaded a global avatar, ``None`` is returned.
        If you want the avatar that a user has displayed, consider :attr:`display_avatar`.
        """
        ...

    @property
    def default_avatar(self) -> Asset:
        """:class:`Asset`: Returns the default avatar for a given user."""
        ...

    @property
    def display_avatar(self) -> Asset:
        """:class:`Asset`: Returns the user's display avatar.

        For regular users this is just their default avatar or uploaded avatar.

        .. versionadded:: 2.0
        """
        ...

    @property
    def banner(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns the user's banner asset, if available.

        .. versionadded:: 2.0


        .. note::
            This information is only available via :meth:`Client.fetch_user`.
        """
        ...

    @property
    def accent_colour(self) -> Optional[Colour]:
        """Optional[:class:`Colour`]: Returns the user's accent colour, if applicable.

        A user's accent colour is only shown if they do not have a banner.
        This will only be available if the user explicitly sets a colour.

        There is an alias for this named :attr:`accent_color`.

        .. versionadded:: 2.0

        .. note::

            This information is only available via :meth:`Client.fetch_user`.
        """
        ...

    @property
    def accent_color(self) -> Optional[Colour]:
        """Optional[:class:`Colour`]: Returns the user's accent color, if applicable.

        A user's accent color is only shown if they do not have a banner.
        This will only be available if the user explicitly sets a color.

        There is an alias for this named :attr:`accent_colour`.

        .. versionadded:: 2.0

        .. note::

            This information is only available via :meth:`Client.fetch_user`.
        """
        ...

    @property
    def colour(self) -> Colour:
        """:class:`Colour`: A property that returns a colour denoting the rendered colour
        for the user. This always returns :meth:`Colour.default`.

        There is an alias for this named :attr:`color`.
        """
        ...

    @property
    def color(self) -> Colour:
        """:class:`Colour`: A property that returns a color denoting the rendered color
        for the user. This always returns :meth:`Colour.default`.

        There is an alias for this named :attr:`colour`.
        """
        ...

    @property
    def mention(self) -> str:
        """:class:`str`: Returns a string that allows you to mention the given user."""
        ...

    @property
    def created_at(self) -> datetime:
        """:class:`datetime.datetime`: Returns the user's creation time in UTC.

        This is when the user's Discord account was created.
        """
        ...

    @property
    def display_name(self) -> str:
        """:class:`str`: Returns the user's display name.

        For regular users this is just their global name or their username,
        but if they have a guild specific nickname then that
        is returned instead.
        """
        ...

    def mentioned_in(self, message: Message) -> bool:
        """Checks if the user is mentioned in the specified message.

        Parameters
        -----------
        message: :class:`Message`
            The message to check if you're mentioned in.

        Returns
        -------
        :class:`bool`
            Indicates if the user is mentioned in the message.
        """
        ...



class ClientUser(BaseUser):
    """Represents your Discord user.

    .. container:: operations

        .. describe:: x == y

            Checks if two users are equal.

        .. describe:: x != y

            Checks if two users are not equal.

        .. describe:: hash(x)

            Return the user's hash.

        .. describe:: str(x)

            Returns the user's handle (e.g. ``name`` or ``name#discriminator``).

    Attributes
    -----------
    name: :class:`str`
        The user's username.
    id: :class:`int`
        The user's unique ID.
    discriminator: :class:`str`
        The user's discriminator. This is a legacy concept that is no longer used.
    global_name: Optional[:class:`str`]
        The user's global nickname, taking precedence over the username in display.

        .. versionadded:: 2.3
    bot: :class:`bool`
        Specifies if the user is a bot account.
    system: :class:`bool`
        Specifies if the user is a system user (i.e. represents Discord officially).

        .. versionadded:: 1.3

    verified: :class:`bool`
        Specifies if the user's email is verified.
    locale: Optional[:class:`str`]
        The IETF language tag used to identify the language the user is using.
    mfa_enabled: :class:`bool`
        Specifies if the user has MFA turned on and working.
    """
    __slots__ = ...
    if TYPE_CHECKING:
        verified: bool
        locale: Optional[str]
        mfa_enabled: bool
        _flags: int
        ...
    def __init__(self, *, state: ConnectionState, data: UserPayload) -> None:
        ...

    def __repr__(self) -> str:
        ...

    async def edit(self, *, username: str = ..., avatar: Optional[bytes] = ...) -> ClientUser:
        """|coro|

        Edits the current profile of the client.

        .. note::

            To upload an avatar, a :term:`py:bytes-like object` must be passed in that
            represents the image being uploaded. If this is done through a file
            then the file must be opened via ``open('some_filename', 'rb')`` and
            the :term:`py:bytes-like object` is given through the use of ``fp.read()``.

            The only image formats supported for uploading is JPEG and PNG.

        .. versionchanged:: 2.0
            The edit is no longer in-place, instead the newly edited client user is returned.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        -----------
        username: :class:`str`
            The new username you wish to change to.
        avatar: Optional[:class:`bytes`]
            A :term:`py:bytes-like object` representing the image to upload.
            Could be ``None`` to denote no avatar.

        Raises
        ------
        HTTPException
            Editing your profile failed.
        ValueError
            Wrong image format passed for ``avatar``.

        Returns
        ---------
        :class:`ClientUser`
            The newly edited client user.
        """
        ...

    @property
    def mutual_guilds(self) -> List[Guild]:
        """List[:class:`Guild`]: The guilds that the user shares with the client.

        .. note::

            This will only return mutual guilds within the client's internal cache.

        .. versionadded:: 1.7
        """
        ...



class User(BaseUser, discord.abc.Messageable):
    """Represents a Discord user.

    .. container:: operations

        .. describe:: x == y

            Checks if two users are equal.

        .. describe:: x != y

            Checks if two users are not equal.

        .. describe:: hash(x)

            Return the user's hash.

        .. describe:: str(x)

            Returns the user's handle (e.g. ``name`` or ``name#discriminator``).

    Attributes
    -----------
    name: :class:`str`
        The user's username.
    id: :class:`int`
        The user's unique ID.
    discriminator: :class:`str`
        The user's discriminator. This is a legacy concept that is no longer used.
    global_name: Optional[:class:`str`]
        The user's global nickname, taking precedence over the username in display.

        .. versionadded:: 2.3
    bot: :class:`bool`
        Specifies if the user is a bot account.
    system: :class:`bool`
        Specifies if the user is a system user (i.e. represents Discord officially).
    """
    __slots__ = ...
    def __repr__(self) -> str:
        ...

    @property
    def dm_channel(self) -> Optional[DMChannel]:
        """Optional[:class:`DMChannel`]: Returns the channel associated with this user if it exists.

        If this returns ``None``, you can create a DM channel by calling the
        :meth:`create_dm` coroutine function.
        """
        ...

    @property
    def mutual_guilds(self) -> List[Guild]:
        """List[:class:`Guild`]: The guilds that the user shares with the client.

        .. note::

            This will only return mutual guilds within the client's internal cache.

        .. versionadded:: 1.7
        """
        ...

    async def create_dm(self) -> DMChannel:
        """|coro|

        Creates a :class:`DMChannel` with this user.

        This should be rarely called, as this is done transparently for most
        people.

        Returns
        -------
        :class:`.DMChannel`
            The channel that was created.
        """
        ...
