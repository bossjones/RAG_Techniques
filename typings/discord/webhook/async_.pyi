"""
This type stub file was generated by pyright.
"""

import asyncio
import aiohttp
import datetime
from typing import Any, Dict, List, Literal, Optional, Sequence, TYPE_CHECKING, Tuple, Type, TypeVar, Union, overload
from contextvars import ContextVar
from ..message import Attachment, Message
from ..user import BaseUser
from ..flags import MessageFlags
from ..asset import Asset
from ..partial_emoji import PartialEmoji
from ..http import HTTPClient, MultipartParameters, Response, Route
from ..mixins import Hashable
from ..channel import ForumChannel, TextChannel, VoiceChannel
from ..file import File
from typing_extensions import Self
from types import TracebackType
from ..embeds import Embed
from ..client import Client
from ..mentions import AllowedMentions
from ..state import ConnectionState
from ..guild import Guild
from ..emoji import Emoji
from ..abc import Snowflake
from ..ui.view import View
from ..types.webhook import SourceGuild as SourceGuildPayload, Webhook as WebhookPayload
from ..types.message import Message as MessagePayload
from ..types.user import PartialUser as PartialUserPayload, User as UserPayload
from ..types.channel import PartialChannel as PartialChannelPayload
from ..types.emoji import PartialEmoji as PartialEmojiPayload

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
__all__ = ('Webhook', 'WebhookMessage', 'PartialWebhookChannel', 'PartialWebhookGuild')
_log = ...
if TYPE_CHECKING:
    BE = TypeVar('BE', bound=BaseException)
    _State = Union[ConnectionState, '_WebhookState']
MISSING: Any = ...
class AsyncDeferredLock:
    def __init__(self, lock: asyncio.Lock) -> None:
        ...

    async def __aenter__(self) -> Self:
        ...

    def delay_by(self, delta: float) -> None:
        ...

    async def __aexit__(self, exc_type: Optional[Type[BE]], exc: Optional[BE], traceback: Optional[TracebackType]) -> None:
        ...



class AsyncWebhookAdapter:
    def __init__(self) -> None:
        ...

    async def request(self, route: Route, session: aiohttp.ClientSession, *, payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., files: Optional[Sequence[File]] = ..., reason: Optional[str] = ..., auth_token: Optional[str] = ..., params: Optional[Dict[str, Any]] = ...) -> Any:
        ...

    def delete_webhook(self, webhook_id: int, *, token: Optional[str] = ..., session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., reason: Optional[str] = ...) -> Response[None]:
        ...

    def delete_webhook_with_token(self, webhook_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., reason: Optional[str] = ...) -> Response[None]:
        ...

    def edit_webhook(self, webhook_id: int, token: str, payload: Dict[str, Any], *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., reason: Optional[str] = ...) -> Response[WebhookPayload]:
        ...

    def edit_webhook_with_token(self, webhook_id: int, token: str, payload: Dict[str, Any], *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., reason: Optional[str] = ...) -> Response[WebhookPayload]:
        ...

    def execute_webhook(self, webhook_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ..., thread_id: Optional[int] = ..., wait: bool = ...) -> Response[Optional[MessagePayload]]:
        ...

    def get_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., thread_id: Optional[int] = ...) -> Response[MessagePayload]:
        ...

    def edit_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ..., thread_id: Optional[int] = ...) -> Response[Message]:
        ...

    def delete_webhook_message(self, webhook_id: int, token: str, message_id: int, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., thread_id: Optional[int] = ...) -> Response[None]:
        ...

    def fetch_webhook(self, webhook_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ...) -> Response[WebhookPayload]:
        ...

    def fetch_webhook_with_token(self, webhook_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ...) -> Response[WebhookPayload]:
        ...

    def create_interaction_response(self, interaction_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., params: MultipartParameters) -> Response[None]:
        ...

    def get_original_interaction_response(self, application_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ...) -> Response[MessagePayload]:
        ...

    def edit_original_interaction_response(self, application_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ..., payload: Optional[Dict[str, Any]] = ..., multipart: Optional[List[Dict[str, Any]]] = ..., files: Optional[Sequence[File]] = ...) -> Response[MessagePayload]:
        ...

    def delete_original_interaction_response(self, application_id: int, token: str, *, session: aiohttp.ClientSession, proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ...) -> Response[None]:
        ...



def interaction_response_params(type: int, data: Optional[Dict[str, Any]] = ...) -> MultipartParameters:
    ...

def interaction_message_response_params(*, type: int, content: Optional[str] = ..., tts: bool = ..., flags: MessageFlags = ..., file: File = ..., files: Sequence[File] = ..., embed: Optional[Embed] = ..., embeds: Sequence[Embed] = ..., attachments: Sequence[Union[Attachment, File]] = ..., view: Optional[View] = ..., allowed_mentions: Optional[AllowedMentions] = ..., previous_allowed_mentions: Optional[AllowedMentions] = ...) -> MultipartParameters:
    ...

async_context: ContextVar[AsyncWebhookAdapter] = ...
class PartialWebhookChannel(Hashable):
    """Represents a partial channel for webhooks.

    These are typically given for channel follower webhooks.

    .. versionadded:: 2.0

    Attributes
    -----------
    id: :class:`int`
        The partial channel's ID.
    name: :class:`str`
        The partial channel's name.
    """
    __slots__ = ...
    def __init__(self, *, data: PartialChannelPayload) -> None:
        ...

    def __repr__(self) -> str:
        ...



class PartialWebhookGuild(Hashable):
    """Represents a partial guild for webhooks.

    These are typically given for channel follower webhooks.

    .. versionadded:: 2.0

    Attributes
    -----------
    id: :class:`int`
        The partial guild's ID.
    name: :class:`str`
        The partial guild's name.
    """
    __slots__ = ...
    def __init__(self, *, data: SourceGuildPayload, state: _State) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def icon(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns the guild's icon asset, if available."""
        ...



class _FriendlyHttpAttributeErrorHelper:
    __slots__ = ...
    def __getattr__(self, attr: str) -> Any:
        ...



class _WebhookState:
    __slots__ = ...
    def __init__(self, webhook: Any, parent: Optional[_State], thread: Snowflake = ...) -> None:
        ...

    def store_user(self, data: Union[UserPayload, PartialUserPayload], *, cache: bool = ...) -> BaseUser:
        ...

    def create_user(self, data: Union[UserPayload, PartialUserPayload]) -> BaseUser:
        ...

    @property
    def allowed_mentions(self) -> Optional[AllowedMentions]:
        ...

    def get_reaction_emoji(self, data: PartialEmojiPayload) -> Union[PartialEmoji, Emoji, str]:
        ...

    @property
    def http(self) -> Union[HTTPClient, _FriendlyHttpAttributeErrorHelper]:
        ...

    def __getattr__(self, attr: str) -> Any:
        ...



class WebhookMessage(Message):
    """Represents a message sent from your webhook.

    This allows you to edit or delete a message sent by your
    webhook.

    This inherits from :class:`discord.Message` with changes to
    :meth:`edit` and :meth:`delete` to work.

    .. versionadded:: 1.6
    """
    _state: _WebhookState
    async def edit(self, *, content: Optional[str] = ..., embeds: Sequence[Embed] = ..., embed: Optional[Embed] = ..., attachments: Sequence[Union[Attachment, File]] = ..., view: Optional[View] = ..., allowed_mentions: Optional[AllowedMentions] = ...) -> WebhookMessage:
        """|coro|

        Edits the message.

        .. versionadded:: 1.6

        .. versionchanged:: 2.0
            The edit is no longer in-place, instead the newly edited message is returned.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        content: Optional[:class:`str`]
            The content to edit the message with or ``None`` to clear it.
        embeds: List[:class:`Embed`]
            A list of embeds to edit the message with.
        embed: Optional[:class:`Embed`]
            The embed to edit the message with. ``None`` suppresses the embeds.
            This should not be mixed with the ``embeds`` parameter.
        attachments: List[Union[:class:`Attachment`, :class:`File`]]
            A list of attachments to keep in the message as well as new files to upload. If ``[]`` is passed
            then all attachments are removed.

            .. note::

                New files will always appear after current attachments.

            .. versionadded:: 2.0
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.
            See :meth:`.abc.Messageable.send` for more information.
        view: Optional[:class:`~discord.ui.View`]
            The updated view to update this message with. If ``None`` is passed then
            the view is removed.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Edited a message that is not yours.
        TypeError
            You specified both ``embed`` and ``embeds``
        ValueError
            The length of ``embeds`` was invalid or
            there was no token associated with this webhook.

        Returns
        --------
        :class:`WebhookMessage`
            The newly edited message.
        """
        ...

    async def add_files(self, *files: File) -> WebhookMessage:
        r"""|coro|

        Adds new files to the end of the message attachments.

        .. versionadded:: 2.0

        Parameters
        -----------
        \*files: :class:`File`
            New files to add to the message.

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Tried to edit a message that isn't yours.

        Returns
        --------
        :class:`WebhookMessage`
            The newly edited message.
        """
        ...

    async def remove_attachments(self, *attachments: Attachment) -> WebhookMessage:
        r"""|coro|

        Removes attachments from the message.

        .. versionadded:: 2.0

        Parameters
        -----------
        \*attachments: :class:`Attachment`
            Attachments to remove from the message.

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Tried to edit a message that isn't yours.

        Returns
        --------
        :class:`WebhookMessage`
            The newly edited message.
        """
        ...

    async def delete(self, *, delay: Optional[float] = ...) -> None:
        """|coro|

        Deletes the message.

        Parameters
        -----------
        delay: Optional[:class:`float`]
            If provided, the number of seconds to wait before deleting the message.
            The waiting is done in the background and deletion failures are ignored.

        Raises
        ------
        Forbidden
            You do not have proper permissions to delete the message.
        NotFound
            The message was deleted already.
        HTTPException
            Deleting the message failed.
        """
        ...



class BaseWebhook(Hashable):
    __slots__: Tuple[str, ...] = ...
    def __init__(self, data: WebhookPayload, token: Optional[str] = ..., state: Optional[_State] = ...) -> None:
        ...

    def is_partial(self) -> bool:
        """:class:`bool`: Whether the webhook is a "partial" webhook.

        .. versionadded:: 2.0"""
        ...

    def is_authenticated(self) -> bool:
        """:class:`bool`: Whether the webhook is authenticated with a bot token.

        .. versionadded:: 2.0
        """
        ...

    @property
    def guild(self) -> Optional[Guild]:
        """Optional[:class:`Guild`]: The guild this webhook belongs to.

        If this is a partial webhook, then this will always return ``None``.
        """
        ...

    @property
    def channel(self) -> Optional[Union[ForumChannel, VoiceChannel, TextChannel]]:
        """Optional[Union[:class:`ForumChannel`, :class:`VoiceChannel`, :class:`TextChannel`]]: The channel this webhook belongs to.

        If this is a partial webhook, then this will always return ``None``.
        """
        ...

    @property
    def created_at(self) -> datetime.datetime:
        """:class:`datetime.datetime`: Returns the webhook's creation time in UTC."""
        ...

    @property
    def avatar(self) -> Optional[Asset]:
        """Optional[:class:`Asset`]: Returns an :class:`Asset` for the avatar the webhook has.

        If the webhook does not have a traditional avatar, ``None`` is returned.
        If you want the avatar that a webhook has displayed, consider :attr:`display_avatar`.
        """
        ...

    @property
    def default_avatar(self) -> Asset:
        """
        :class:`Asset`: Returns the default avatar. This is always the blurple avatar.

        .. versionadded:: 2.0
        """
        ...

    @property
    def display_avatar(self) -> Asset:
        """:class:`Asset`: Returns the webhook's display avatar.

        This is either webhook's default avatar or uploaded avatar.

        .. versionadded:: 2.0
        """
        ...



class Webhook(BaseWebhook):
    """Represents an asynchronous Discord webhook.

    Webhooks are a form to send messages to channels in Discord without a
    bot user or authentication.

    There are two main ways to use Webhooks. The first is through the ones
    received by the library such as :meth:`.Guild.webhooks`,
    :meth:`.TextChannel.webhooks`, :meth:`.VoiceChannel.webhooks`
    and :meth:`.ForumChannel.webhooks`.
    The ones received by the library will automatically be
    bound using the library's internal HTTP session.

    The second form involves creating a webhook object manually using the
    :meth:`~.Webhook.from_url` or :meth:`~.Webhook.partial` classmethods.

    For example, creating a webhook from a URL and using :doc:`aiohttp <aio:index>`:

    .. code-block:: python3

        from discord import Webhook
        import aiohttp

        async def foo():
            async with aiohttp.ClientSession() as session:
                webhook = Webhook.from_url('url-here', session=session)
                await webhook.send('Hello World', username='Foo')

    For a synchronous counterpart, see :class:`SyncWebhook`.

    .. container:: operations

        .. describe:: x == y

            Checks if two webhooks are equal.

        .. describe:: x != y

            Checks if two webhooks are not equal.

        .. describe:: hash(x)

            Returns the webhooks's hash.

    .. versionchanged:: 1.4
        Webhooks are now comparable and hashable.

    Attributes
    ------------
    id: :class:`int`
        The webhook's ID
    type: :class:`WebhookType`
        The type of the webhook.

        .. versionadded:: 1.3

    token: Optional[:class:`str`]
        The authentication token of the webhook. If this is ``None``
        then the webhook cannot be used to make requests.
    guild_id: Optional[:class:`int`]
        The guild ID this webhook is for.
    channel_id: Optional[:class:`int`]
        The channel ID this webhook is for.
    user: Optional[:class:`abc.User`]
        The user this webhook was created by. If the webhook was
        received without authentication then this will be ``None``.
    name: Optional[:class:`str`]
        The default name of the webhook.
    source_guild: Optional[:class:`PartialWebhookGuild`]
        The guild of the channel that this webhook is following.
        Only given if :attr:`type` is :attr:`WebhookType.channel_follower`.

        .. versionadded:: 2.0

    source_channel: Optional[:class:`PartialWebhookChannel`]
        The channel that this webhook is following.
        Only given if :attr:`type` is :attr:`WebhookType.channel_follower`.

        .. versionadded:: 2.0
    """
    __slots__: Tuple[str, ...] = ...
    def __init__(self, data: WebhookPayload, session: aiohttp.ClientSession, token: Optional[str] = ..., state: Optional[_State] = ..., proxy: Optional[str] = ..., proxy_auth: Optional[aiohttp.BasicAuth] = ...) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @property
    def url(self) -> str:
        """:class:`str` : Returns the webhook's url."""
        ...

    @classmethod
    def partial(cls, id: int, token: str, *, session: aiohttp.ClientSession = ..., client: Client = ..., bot_token: Optional[str] = ...) -> Self:
        """Creates a partial :class:`Webhook`.

        Parameters
        -----------
        id: :class:`int`
            The ID of the webhook.
        token: :class:`str`
            The authentication token of the webhook.
        session: :class:`aiohttp.ClientSession`
            The session to use to send requests with. Note
            that the library does not manage the session and
            will not close it.

            .. versionadded:: 2.0
        client: :class:`Client`
            The client to initialise this webhook with. This allows it to
            attach the client's internal state. If ``session`` is not given
            while this is given then the client's internal session will be used.

            .. versionadded:: 2.2
        bot_token: Optional[:class:`str`]
            The bot authentication token for authenticated requests
            involving the webhook.

            .. versionadded:: 2.0

        Raises
        -------
        TypeError
            Neither ``session`` nor ``client`` were given.

        Returns
        --------
        :class:`Webhook`
            A partial :class:`Webhook`.
            A partial webhook is just a webhook object with an ID and a token.
        """
        ...

    @classmethod
    def from_url(cls, url: str, *, session: aiohttp.ClientSession = ..., client: Client = ..., bot_token: Optional[str] = ...) -> Self:
        """Creates a partial :class:`Webhook` from a webhook URL.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        url: :class:`str`
            The URL of the webhook.
        session: :class:`aiohttp.ClientSession`
            The session to use to send requests with. Note
            that the library does not manage the session and
            will not close it.

            .. versionadded:: 2.0
        client: :class:`Client`
            The client to initialise this webhook with. This allows it to
            attach the client's internal state. If ``session`` is not given
            while this is given then the client's internal session will be used.

            .. versionadded:: 2.2
        bot_token: Optional[:class:`str`]
            The bot authentication token for authenticated requests
            involving the webhook.

            .. versionadded:: 2.0

        Raises
        -------
        ValueError
            The URL is invalid.
        TypeError
            Neither ``session`` nor ``client`` were given.

        Returns
        --------
        :class:`Webhook`
            A partial :class:`Webhook`.
            A partial webhook is just a webhook object with an ID and a token.
        """
        ...

    @classmethod
    def from_state(cls, data: WebhookPayload, state: ConnectionState) -> Self:
        ...

    async def fetch(self, *, prefer_auth: bool = ...) -> Webhook:
        """|coro|

        Fetches the current webhook.

        This could be used to get a full webhook from a partial webhook.

        .. versionadded:: 2.0

        .. note::

            When fetching with an unauthenticated webhook, i.e.
            :meth:`is_authenticated` returns ``False``, then the
            returned webhook does not contain any user information.

        Parameters
        -----------
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

        Raises
        -------
        HTTPException
            Could not fetch the webhook
        NotFound
            Could not find the webhook by this ID
        ValueError
            This webhook does not have a token associated with it.

        Returns
        --------
        :class:`Webhook`
            The fetched webhook.
        """
        ...

    async def delete(self, *, reason: Optional[str] = ..., prefer_auth: bool = ...) -> None:
        """|coro|

        Deletes this Webhook.

        Parameters
        ------------
        reason: Optional[:class:`str`]
            The reason for deleting this webhook. Shows up on the audit log.

            .. versionadded:: 1.4
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Deleting the webhook failed.
        NotFound
            This webhook does not exist.
        Forbidden
            You do not have permissions to delete this webhook.
        ValueError
            This webhook does not have a token associated with it.
        """
        ...

    async def edit(self, *, reason: Optional[str] = ..., name: Optional[str] = ..., avatar: Optional[bytes] = ..., channel: Optional[Snowflake] = ..., prefer_auth: bool = ...) -> Webhook:
        """|coro|

        Edits this Webhook.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        name: Optional[:class:`str`]
            The webhook's new default name.
        avatar: Optional[:class:`bytes`]
            A :term:`py:bytes-like object` representing the webhook's new default avatar.
        channel: Optional[:class:`abc.Snowflake`]
            The webhook's new channel. This requires an authenticated webhook.

            .. versionadded:: 2.0
        reason: Optional[:class:`str`]
            The reason for editing this webhook. Shows up on the audit log.

            .. versionadded:: 1.4
        prefer_auth: :class:`bool`
            Whether to use the bot token over the webhook token
            if available. Defaults to ``True``.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Editing the webhook failed.
        NotFound
            This webhook does not exist.
        ValueError
            This webhook does not have a token associated with it
            or it tried editing a channel without authentication.
        """
        ...

    @overload
    async def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., ephemeral: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., view: View = ..., thread: Snowflake = ..., thread_name: str = ..., wait: Literal[True], suppress_embeds: bool = ..., silent: bool = ...) -> WebhookMessage:
        ...

    @overload
    async def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., ephemeral: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., view: View = ..., thread: Snowflake = ..., thread_name: str = ..., wait: Literal[False] = ..., suppress_embeds: bool = ..., silent: bool = ...) -> None:
        ...

    async def send(self, content: str = ..., *, username: str = ..., avatar_url: Any = ..., tts: bool = ..., ephemeral: bool = ..., file: File = ..., files: Sequence[File] = ..., embed: Embed = ..., embeds: Sequence[Embed] = ..., allowed_mentions: AllowedMentions = ..., view: View = ..., thread: Snowflake = ..., thread_name: str = ..., wait: bool = ..., suppress_embeds: bool = ..., silent: bool = ...) -> Optional[WebhookMessage]:
        """|coro|

        Sends a message using the webhook.

        The content must be a type that can convert to a string through ``str(content)``.

        To upload a single file, the ``file`` parameter should be used with a
        single :class:`File` object.

        If the ``embed`` parameter is provided, it must be of type :class:`Embed` and
        it must be a rich embed type. You cannot mix the ``embed`` parameter with the
        ``embeds`` parameter, which must be a :class:`list` of :class:`Embed` objects to send.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        content: :class:`str`
            The content of the message to send.
        wait: :class:`bool`
            Whether the server should wait before sending a response. This essentially
            means that the return type of this function changes from ``None`` to
            a :class:`WebhookMessage` if set to ``True``. If the type of webhook
            is :attr:`WebhookType.application` then this is always set to ``True``.
        username: :class:`str`
            The username to send with this message. If no username is provided
            then the default username for the webhook is used.
        avatar_url: :class:`str`
            The avatar URL to send with this message. If no avatar URL is provided
            then the default avatar for the webhook is used. If this is not a
            string then it is explicitly cast using ``str``.
        tts: :class:`bool`
            Indicates if the message should be sent using text-to-speech.
        ephemeral: :class:`bool`
            Indicates if the message should only be visible to the user.
            This is only available to :attr:`WebhookType.application` webhooks.
            If a view is sent with an ephemeral message and it has no timeout set
            then the timeout is set to 15 minutes.

            .. versionadded:: 2.0
        file: :class:`File`
            The file to upload. This cannot be mixed with ``files`` parameter.
        files: List[:class:`File`]
            A list of files to send with the content. This cannot be mixed with the
            ``file`` parameter.
        embed: :class:`Embed`
            The rich embed for the content to send. This cannot be mixed with
            ``embeds`` parameter.
        embeds: List[:class:`Embed`]
            A list of embeds to send with the content. Maximum of 10. This cannot
            be mixed with the ``embed`` parameter.
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.

            .. versionadded:: 1.4
        view: :class:`discord.ui.View`
            The view to send with the message. You can only send a view
            if this webhook is not partial and has state attached. A
            webhook has state attached if the webhook is managed by the
            library.

            .. versionadded:: 2.0
        thread: :class:`~discord.abc.Snowflake`
            The thread to send this webhook to.

            .. versionadded:: 2.0
        thread_name: :class:`str`
            The thread name to create with this webhook if the webhook belongs
            to a :class:`~discord.ForumChannel`. Note that this is mutually
            exclusive with the ``thread`` parameter, as this will create a
            new thread with the given name.

            .. versionadded:: 2.0
        suppress_embeds: :class:`bool`
            Whether to suppress embeds for the message. This sends the message without any embeds if set to ``True``.

            .. versionadded:: 2.0
        silent: :class:`bool`
            Whether to suppress push and desktop notifications for the message. This will increment the mention counter
            in the UI, but will not actually send a notification.

            .. versionadded:: 2.2

        Raises
        --------
        HTTPException
            Sending the message failed.
        NotFound
            This webhook was not found.
        Forbidden
            The authorization token for the webhook is incorrect.
        TypeError
            You specified both ``embed`` and ``embeds`` or ``file`` and ``files``
            or ``thread`` and ``thread_name``.
        ValueError
            The length of ``embeds`` was invalid, there was no token
            associated with this webhook or ``ephemeral`` was passed
            with the improper webhook type or there was no state
            attached with this webhook when giving it a view.

        Returns
        ---------
        Optional[:class:`WebhookMessage`]
            If ``wait`` is ``True`` then the message that was sent, otherwise ``None``.
        """
        ...

    async def fetch_message(self, id: int, /, *, thread: Snowflake = ...) -> WebhookMessage:
        """|coro|

        Retrieves a single :class:`~discord.WebhookMessage` owned by this webhook.

        .. versionadded:: 2.0

        Parameters
        ------------
        id: :class:`int`
            The message ID to look for.
        thread: :class:`~discord.abc.Snowflake`
            The thread to look in.

        Raises
        --------
        ~discord.NotFound
            The specified message was not found.
        ~discord.Forbidden
            You do not have the permissions required to get a message.
        ~discord.HTTPException
            Retrieving the message failed.
        ValueError
            There was no token associated with this webhook.

        Returns
        --------
        :class:`~discord.WebhookMessage`
            The message asked for.
        """
        ...

    async def edit_message(self, message_id: int, *, content: Optional[str] = ..., embeds: Sequence[Embed] = ..., embed: Optional[Embed] = ..., attachments: Sequence[Union[Attachment, File]] = ..., view: Optional[View] = ..., allowed_mentions: Optional[AllowedMentions] = ..., thread: Snowflake = ...) -> WebhookMessage:
        """|coro|

        Edits a message owned by this webhook.

        This is a lower level interface to :meth:`WebhookMessage.edit` in case
        you only have an ID.

        .. versionadded:: 1.6

        .. versionchanged:: 2.0
            The edit is no longer in-place, instead the newly edited message is returned.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        message_id: :class:`int`
            The message ID to edit.
        content: Optional[:class:`str`]
            The content to edit the message with or ``None`` to clear it.
        embeds: List[:class:`Embed`]
            A list of embeds to edit the message with.
        embed: Optional[:class:`Embed`]
            The embed to edit the message with. ``None`` suppresses the embeds.
            This should not be mixed with the ``embeds`` parameter.
        attachments: List[Union[:class:`Attachment`, :class:`File`]]
            A list of attachments to keep in the message as well as new files to upload. If ``[]`` is passed
            then all attachments are removed.

            .. versionadded:: 2.0
        allowed_mentions: :class:`AllowedMentions`
            Controls the mentions being processed in this message.
            See :meth:`.abc.Messageable.send` for more information.
        view: Optional[:class:`~discord.ui.View`]
            The updated view to update this message with. If ``None`` is passed then
            the view is removed. The webhook must have state attached, similar to
            :meth:`send`.

            .. versionadded:: 2.0
        thread: :class:`~discord.abc.Snowflake`
            The thread the webhook message belongs to.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Editing the message failed.
        Forbidden
            Edited a message that is not yours.
        TypeError
            You specified both ``embed`` and ``embeds``
        ValueError
            The length of ``embeds`` was invalid,
            there was no token associated with this webhook or the webhook had
            no state.

        Returns
        --------
        :class:`WebhookMessage`
            The newly edited webhook message.
        """
        ...

    async def delete_message(self, message_id: int, /, *, thread: Snowflake = ...) -> None:
        """|coro|

        Deletes a message owned by this webhook.

        This is a lower level interface to :meth:`WebhookMessage.delete` in case
        you only have an ID.

        .. versionadded:: 1.6

        .. versionchanged:: 2.0

            ``message_id`` parameter is now positional-only.

        .. versionchanged:: 2.0
            This function will now raise :exc:`ValueError` instead of
            ``InvalidArgument``.

        Parameters
        ------------
        message_id: :class:`int`
            The message ID to delete.
        thread: :class:`~discord.abc.Snowflake`
            The thread the webhook message belongs to.

            .. versionadded:: 2.0

        Raises
        -------
        HTTPException
            Deleting the message failed.
        Forbidden
            Deleted a message that is not yours.
        ValueError
            This webhook does not have a token associated with it.
        """
        ...
