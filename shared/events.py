"""Simple events, C# style"""

import typing

class Event:
    """Describes an event, which has a particular name and an optional handler style.

    Attributes:
        name (str): a name for this event, used for error-handling
        subscribers (list): a list of callables which subscribe to this event. After invoking
            each one, the result is passed to the handler if there is one. The very last result
            is returned.
        handler (callable(args, kwargs, result) -> cont, args, kwargs): called with the result of
            the subscriber and must return the new arguments and keyword-arguments. if cont is
            false, future events are not called
    """

    def __init__(self, name: str, handler: typing.Optional[typing.Callable] = None):
        self.name = name
        self.subscribers = []
        self.handler = handler

    def __iadd__(self, subscriber: typing.Callable):
        self.subscribers.append(subscriber)
        return self

    def __isub__(self, subscriber: typing.Callable):
        self.subscribers.remove(subscriber)
        return self

    def __call__(self, *args, **kwargs):
        result = None
        for subscriber in self.subscribers:
            result = subscriber(*args, **kwargs)
            if self.handler:
                cont, args, kwargs = self.handler(args, kwargs, result)
                if not cont:
                    break
        return result

    def __repr__(self):
        _handler = 'has' if self.handler is not None else 'no'
        return f'Event \'{self.name}\' ({len(self.subscribers)} subscribers, {_handler} handler)'
