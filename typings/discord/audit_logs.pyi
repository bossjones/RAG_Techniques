"""
This type stub file was generated by pyright.
"""

import datetime
from typing import Any, Callable, ClassVar, Generator, List, Mapping, Optional, TYPE_CHECKING, Tuple, TypeVar, Union
from . import abc, enums, flags, utils
from .invite import Invite
from .mixins import Hashable
from .object import Object
from .automod import AutoModRule
from .role import Role
from .emoji import Emoji
from .member import Member
from .scheduled_event import ScheduledEvent
from .stage_instance import StageInstance
from .sticker import GuildSticker
from .threads import Thread
from .integrations import PartialIntegration
from .guild import Guild
from .types.audit_log import AuditLogChange as AuditLogChangePayload, AuditLogEntry as AuditLogEntryPayload
from .user import User
from .app_commands import AppCommand
from .webhook import Webhook

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
__all__ = ('AuditLogDiff', 'AuditLogChanges', 'AuditLogEntry')
if TYPE_CHECKING:
    TargetType = Union[Guild, abc.GuildChannel, Member, User, Role, Invite, Emoji, StageInstance, GuildSticker, Thread, Object, PartialIntegration, AutoModRule, ScheduledEvent, Webhook, AppCommand, None,]
E = TypeVar('E', bound=enums.Enum)
F = TypeVar('F', bound=flags.BaseFlags)
class AuditLogDiff:
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Generator[Tuple[str, Any], None, None]:
        ...

    def __repr__(self) -> str:
        ...

    if TYPE_CHECKING:
        def __getattr__(self, item: str) -> Any:
            ...

        def __setattr__(self, key: str, value: Any) -> Any:
            ...



Transformer = Callable[["AuditLogEntry", Any], Any]
class AuditLogChanges:
    TRANSFORMERS: ClassVar[Mapping[str, Tuple[Optional[str], Optional[Transformer]]]] = ...
    def __init__(self, entry: AuditLogEntry, data: List[AuditLogChangePayload]) -> None:
        ...

    def __repr__(self) -> str:
        ...



class _AuditLogProxy:
    def __init__(self, **kwargs: Any) -> None:
        ...



class _AuditLogProxyMemberPrune(_AuditLogProxy):
    delete_member_days: int
    members_removed: int
    ...


class _AuditLogProxyMemberMoveOrMessageDelete(_AuditLogProxy):
    channel: Union[abc.GuildChannel, Thread]
    count: int
    ...


class _AuditLogProxyMemberDisconnect(_AuditLogProxy):
    count: int
    ...


class _AuditLogProxyPinAction(_AuditLogProxy):
    channel: Union[abc.GuildChannel, Thread]
    message_id: int
    ...


class _AuditLogProxyStageInstanceAction(_AuditLogProxy):
    channel: abc.GuildChannel
    ...


class _AuditLogProxyMessageBulkDelete(_AuditLogProxy):
    count: int
    ...


class _AuditLogProxyAutoModAction(_AuditLogProxy):
    automod_rule_name: str
    automod_rule_trigger_type: str
    channel: Optional[Union[abc.GuildChannel, Thread]]
    ...


class AuditLogEntry(Hashable):
    r"""Represents an Audit Log entry.

    You retrieve these via :meth:`Guild.audit_logs`.

    .. container:: operations

        .. describe:: x == y

            Checks if two entries are equal.

        .. describe:: x != y

            Checks if two entries are not equal.

        .. describe:: hash(x)

            Returns the entry's hash.

    .. versionchanged:: 1.7
        Audit log entries are now comparable and hashable.

    Attributes
    -----------
    action: :class:`AuditLogAction`
        The action that was done.
    user: Optional[:class:`abc.User`]
        The user who initiated this action. Usually a :class:`Member`\, unless gone
        then it's a :class:`User`.
    user_id: Optional[:class:`int`]
        The user ID who initiated this action.

        .. versionadded:: 2.2
    id: :class:`int`
        The entry ID.
    guild: :class:`Guild`
        The guild that this entry belongs to.
    target: Any
        The target that got changed. The exact type of this depends on
        the action being done.
    reason: Optional[:class:`str`]
        The reason this action was done.
    extra: Any
        Extra information that this entry has that might be useful.
        For most actions, this is ``None``. However in some cases it
        contains extra information. See :class:`AuditLogAction` for
        which actions have this field filled out.
    """
    def __init__(self, *, users: Mapping[int, User], integrations: Mapping[int, PartialIntegration], app_commands: Mapping[int, AppCommand], automod_rules: Mapping[int, AutoModRule], webhooks: Mapping[int, Webhook], data: AuditLogEntryPayload, guild: Guild) -> None:
        ...

    def __repr__(self) -> str:
        ...

    @utils.cached_property
    def created_at(self) -> datetime.datetime:
        """:class:`datetime.datetime`: Returns the entry's creation time in UTC."""
        ...

    @utils.cached_property
    def target(self) -> TargetType:
        ...

    @utils.cached_property
    def category(self) -> Optional[enums.AuditLogActionCategory]:
        """Optional[:class:`AuditLogActionCategory`]: The category of the action, if applicable."""
        ...

    @utils.cached_property
    def changes(self) -> AuditLogChanges:
        """:class:`AuditLogChanges`: The list of changes this entry has."""
        ...

    @utils.cached_property
    def before(self) -> AuditLogDiff:
        """:class:`AuditLogDiff`: The target's prior state."""
        ...

    @utils.cached_property
    def after(self) -> AuditLogDiff:
        """:class:`AuditLogDiff`: The target's subsequent state."""
        ...
