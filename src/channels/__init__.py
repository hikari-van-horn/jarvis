from abc import ABC, abstractmethod


class BaseChannel(ABC):
    """Base class for communication channels."""

    @abstractmethod
    async def start(self):
        """Start the channel's event loop."""
        pass

