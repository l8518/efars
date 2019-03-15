from abc import ABC, abstractmethod

class AbstractReceiverAdapter(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fetch_user(self, user, rating_n):
        pass



