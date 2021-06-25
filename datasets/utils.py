from dataclasses import dataclass


@dataclass
class ImageShape:
    channels: int
    width: int
    height: int

    def __iter__(self):
        # making this class iterable means we can convert it to a list if we need to
        for each in self.__dict__.values():
            yield each
