from abc import ABC, abstractmethod


class Stream(ABC):
    @abstractmethod
    def start(self):
        """Start the video stream."""

    @abstractmethod
    def update(self):
        """Update the frame."""

    @abstractmethod
    def read(self):
        """Return the current frame."""

    @abstractmethod
    def stop(self):
        """Stop the video stream."""