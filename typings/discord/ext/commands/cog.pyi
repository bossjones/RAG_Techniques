"""
This type stub file was generated by pyright.
"""

import discord
from discord import app_commands
from typing import Any, Callable, ClassVar, Coroutine, Dict, Generator, List, Optional, TYPE_CHECKING, Tuple, TypeVar, Union
from ._types import BotT
from typing_extensions import Self
from discord._types import ClientT
from .context import Context
from .core import Command

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
__all__ = ('CogMeta', 'Cog', 'GroupCog')
FuncT = TypeVar('FuncT', bound=Callable[..., Any])
MISSING: Any = ...
class CogMeta(type):
    """A metaclass for defining a cog.

    Note that you should probably not use this directly. It is exposed
    purely for documentation purposes along with making custom metaclasses to intermix
    with other metaclasses such as the :class:`abc.ABCMeta` metaclass.

    For example, to create an abstract cog mixin class, the following would be done.

    .. code-block:: python3

        import abc

        class CogABCMeta(commands.CogMeta, abc.ABCMeta):
            pass

        class SomeMixin(metaclass=abc.ABCMeta):
            pass

        class SomeCogMixin(SomeMixin, commands.Cog, metaclass=CogABCMeta):
            pass

    .. note::

        When passing an attribute of a metaclass that is documented below, note
        that you must pass it as a keyword-only argument to the class creation
        like the following example:

        .. code-block:: python3

            class MyCog(commands.Cog, name='My Cog'):
                pass

    Attributes
    -----------
    name: :class:`str`
        The cog name. By default, it is the name of the class with no modification.
    description: :class:`str`
        The cog description. By default, it is the cleaned docstring of the class.

        .. versionadded:: 1.6

    command_attrs: :class:`dict`
        A list of attributes to apply to every command inside this cog. The dictionary
        is passed into the :class:`Command` options at ``__init__``.
        If you specify attributes inside the command attribute in the class, it will
        override the one specified inside this attribute. For example:

        .. code-block:: python3

            class MyCog(commands.Cog, command_attrs=dict(hidden=True)):
                @commands.command()
                async def foo(self, ctx):
                    pass # hidden -> True

                @commands.command(hidden=False)
                async def bar(self, ctx):
                    pass # hidden -> False

    group_name: Union[:class:`str`, :class:`~discord.app_commands.locale_str`]
        The group name of a cog. This is only applicable for :class:`GroupCog` instances.
        By default, it's the same value as :attr:`name`.

        .. versionadded:: 2.0
    group_description: Union[:class:`str`, :class:`~discord.app_commands.locale_str`]
        The group description of a cog. This is only applicable for :class:`GroupCog` instances.
        By default, it's the same value as :attr:`description`.

        .. versionadded:: 2.0
    group_nsfw: :class:`bool`
        Whether the application command group is NSFW. This is only applicable for :class:`GroupCog` instances.
        By default, it's ``False``.

        .. versionadded:: 2.0
    group_auto_locale_strings: :class:`bool`
        If this is set to ``True``, then all translatable strings will implicitly
        be wrapped into :class:`~discord.app_commands.locale_str` rather
        than :class:`str`. Defaults to ``True``.

        .. versionadded:: 2.0
    group_extras: :class:`dict`
        A dictionary that can be used to store extraneous data.
        This is only applicable for :class:`GroupCog` instances.
        The library will not touch any values or keys within this dictionary.

        .. versionadded:: 2.1
    """
    __cog_name__: str
    __cog_description__: str
    __cog_group_name__: Union[str, app_commands.locale_str]
    __cog_group_description__: Union[str, app_commands.locale_str]
    __cog_group_nsfw__: bool
    __cog_group_auto_locale_strings__: bool
    __cog_group_extras__: Dict[Any, Any]
    __cog_settings__: Dict[str, Any]
    __cog_commands__: List[Command[Any, ..., Any]]
    __cog_app_commands__: List[Union[app_commands.Group, app_commands.Command[Any, ..., Any]]]
    __cog_listeners__: List[Tuple[str, str]]
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        ...

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        ...

    @classmethod
    def qualified_name(cls) -> str:
        ...



class Cog(metaclass=CogMeta):
    """The base class that all cogs must inherit from.

    A cog is a collection of commands, listeners, and optional state to
    help group commands together. More information on them can be found on
    the :ref:`ext_commands_cogs` page.

    When inheriting from this class, the options shown in :class:`CogMeta`
    are equally valid here.
    """
    __cog_name__: str
    __cog_description__: str
    __cog_group_name__: Union[str, app_commands.locale_str]
    __cog_group_description__: Union[str, app_commands.locale_str]
    __cog_settings__: Dict[str, Any]
    __cog_commands__: List[Command[Self, ..., Any]]
    __cog_app_commands__: List[Union[app_commands.Group, app_commands.Command[Self, ..., Any]]]
    __cog_listeners__: List[Tuple[str, str]]
    __cog_is_app_commands_group__: ClassVar[bool] = ...
    __cog_app_commands_group__: Optional[app_commands.Group]
    __discord_app_commands_error_handler__: Optional[Callable[[discord.Interaction, app_commands.AppCommandError], Coroutine[Any, Any, None]]]
    def __new__(cls, *args: Any, **kwargs: Any) -> Self:
        ...

    def get_commands(self) -> List[Command[Self, ..., Any]]:
        r"""Returns the commands that are defined inside this cog.

        This does *not* include :class:`discord.app_commands.Command` or :class:`discord.app_commands.Group`
        instances.

        Returns
        --------
        List[:class:`.Command`]
            A :class:`list` of :class:`.Command`\s that are
            defined inside this cog, not including subcommands.
        """
        ...

    def get_app_commands(self) -> List[Union[app_commands.Command[Self, ..., Any], app_commands.Group]]:
        r"""Returns the app commands that are defined inside this cog.

        Returns
        --------
        List[Union[:class:`discord.app_commands.Command`, :class:`discord.app_commands.Group`]]
            A :class:`list` of :class:`discord.app_commands.Command`\s and :class:`discord.app_commands.Group`\s that are
            defined inside this cog, not including subcommands.
        """
        ...

    @property
    def qualified_name(self) -> str:
        """:class:`str`: Returns the cog's specified name, not the class name."""
        ...

    @property
    def description(self) -> str:
        """:class:`str`: Returns the cog's description, typically the cleaned docstring."""
        ...

    @description.setter
    def description(self, description: str) -> None:
        ...

    def walk_commands(self) -> Generator[Command[Self, ..., Any], None, None]:
        """An iterator that recursively walks through this cog's commands and subcommands.

        Yields
        ------
        Union[:class:`.Command`, :class:`.Group`]
            A command or group from the cog.
        """
        ...

    def walk_app_commands(self) -> Generator[Union[app_commands.Command[Self, ..., Any], app_commands.Group], None, None]:
        """An iterator that recursively walks through this cog's app commands and subcommands.

        Yields
        ------
        Union[:class:`discord.app_commands.Command`, :class:`discord.app_commands.Group`]
            An app command or group from the cog.
        """
        ...

    @property
    def app_command(self) -> Optional[app_commands.Group]:
        """Optional[:class:`discord.app_commands.Group`]: Returns the associated group with this cog.

        This is only available if inheriting from :class:`GroupCog`.
        """
        ...

    def get_listeners(self) -> List[Tuple[str, Callable[..., Any]]]:
        """Returns a :class:`list` of (name, function) listener pairs that are defined in this cog.

        Returns
        --------
        List[Tuple[:class:`str`, :ref:`coroutine <coroutine>`]]
            The listeners defined in this cog.
        """
        ...

    @classmethod
    def listener(cls, name: str = ...) -> Callable[[FuncT], FuncT]:
        """A decorator that marks a function as a listener.

        This is the cog equivalent of :meth:`.Bot.listen`.

        Parameters
        ------------
        name: :class:`str`
            The name of the event being listened to. If not provided, it
            defaults to the function's name.

        Raises
        --------
        TypeError
            The function is not a coroutine function or a string was not passed as
            the name.
        """
        ...

    def has_error_handler(self) -> bool:
        """:class:`bool`: Checks whether the cog has an error handler.

        .. versionadded:: 1.7
        """
        ...

    def has_app_command_error_handler(self) -> bool:
        """:class:`bool`: Checks whether the cog has an app error handler.

        .. versionadded:: 2.1
        """
        ...

    @_cog_special_method
    async def cog_load(self) -> None:
        """|maybecoro|

        A special method that is called when the cog gets loaded.

        Subclasses must replace this if they want special asynchronous loading behaviour.
        Note that the ``__init__`` special method does not allow asynchronous code to run
        inside it, thus this is helpful for setting up code that needs to be asynchronous.

        .. versionadded:: 2.0
        """
        ...

    @_cog_special_method
    async def cog_unload(self) -> None:
        """|maybecoro|

        A special method that is called when the cog gets removed.

        Subclasses must replace this if they want special unloading behaviour.

        Exceptions raised in this method are ignored during extension unloading.

        .. versionchanged:: 2.0

            This method can now be a :term:`coroutine`.
        """
        ...

    @_cog_special_method
    def bot_check_once(self, ctx: Context[BotT]) -> bool:
        """A special method that registers as a :meth:`.Bot.check_once`
        check.

        This function **can** be a coroutine and must take a sole parameter,
        ``ctx``, to represent the :class:`.Context`.
        """
        ...

    @_cog_special_method
    def bot_check(self, ctx: Context[BotT]) -> bool:
        """A special method that registers as a :meth:`.Bot.check`
        check.

        This function **can** be a coroutine and must take a sole parameter,
        ``ctx``, to represent the :class:`.Context`.
        """
        ...

    @_cog_special_method
    def cog_check(self, ctx: Context[BotT]) -> bool:
        """A special method that registers as a :func:`~discord.ext.commands.check`
        for every command and subcommand in this cog.

        This function **can** be a coroutine and must take a sole parameter,
        ``ctx``, to represent the :class:`.Context`.
        """
        ...

    @_cog_special_method
    def interaction_check(self, interaction: discord.Interaction[ClientT], /) -> bool:
        """A special method that registers as a :func:`discord.app_commands.check`
        for every app command and subcommand in this cog.

        This function **can** be a coroutine and must take a sole parameter,
        ``interaction``, to represent the :class:`~discord.Interaction`.

        .. versionadded:: 2.0
        """
        ...

    @_cog_special_method
    async def cog_command_error(self, ctx: Context[BotT], error: Exception) -> None:
        """|coro|

        A special method that is called whenever an error
        is dispatched inside this cog.

        This is similar to :func:`.on_command_error` except only applying
        to the commands inside this cog.

        This **must** be a coroutine.

        Parameters
        -----------
        ctx: :class:`.Context`
            The invocation context where the error happened.
        error: :class:`CommandError`
            The error that happened.
        """
        ...

    @_cog_special_method
    async def cog_app_command_error(self, interaction: discord.Interaction, error: app_commands.AppCommandError) -> None:
        """|coro|

        A special method that is called whenever an error within
        an application command is dispatched inside this cog.

        This is similar to :func:`discord.app_commands.CommandTree.on_error` except
        only applying to the application commands inside this cog.

        This **must** be a coroutine.

        Parameters
        -----------
        interaction: :class:`~discord.Interaction`
            The interaction that is being handled.
        error: :exc:`~discord.app_commands.AppCommandError`
            The exception that was raised.
        """
        ...

    @_cog_special_method
    async def cog_before_invoke(self, ctx: Context[BotT]) -> None:
        """|coro|

        A special method that acts as a cog local pre-invoke hook.

        This is similar to :meth:`.Command.before_invoke`.

        This **must** be a coroutine.

        Parameters
        -----------
        ctx: :class:`.Context`
            The invocation context.
        """
        ...

    @_cog_special_method
    async def cog_after_invoke(self, ctx: Context[BotT]) -> None:
        """|coro|

        A special method that acts as a cog local post-invoke hook.

        This is similar to :meth:`.Command.after_invoke`.

        This **must** be a coroutine.

        Parameters
        -----------
        ctx: :class:`.Context`
            The invocation context.
        """
        ...



class GroupCog(Cog):
    """Represents a cog that also doubles as a parent :class:`discord.app_commands.Group` for
    the application commands defined within it.

    This inherits from :class:`Cog` and the options in :class:`CogMeta` also apply to this.
    See the :class:`Cog` documentation for methods.

    Decorators such as :func:`~discord.app_commands.guild_only`, :func:`~discord.app_commands.guilds`,
    and :func:`~discord.app_commands.default_permissions` will apply to the group if used on top of the
    cog.

    Hybrid commands will also be added to the Group, giving the ability to categorize slash commands into
    groups, while keeping the prefix-style command as a root-level command.

    For example:

    .. code-block:: python3

        from discord import app_commands
        from discord.ext import commands

        @app_commands.guild_only()
        class MyCog(commands.GroupCog, group_name='my-cog'):
            pass

    .. versionadded:: 2.0
    """
    __cog_is_app_commands_group__: ClassVar[bool] = ...
