from abc import ABC, abstractmethod

class AbstractProviderAdapter(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def emit_rating(self, user_id, item_id, rating):
        pass
