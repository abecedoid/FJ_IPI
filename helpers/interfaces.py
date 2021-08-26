from abc import ABC, abstractmethod


class ParticlePosition(ABC):

    @abstractmethod
    def center(self):
        pass

    @abstractmethod
    def radius(self):
        pass