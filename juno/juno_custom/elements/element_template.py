from abc import ABC, abstractmethod

# from juno import Lens


class ElementTemplate(ABC):
    def __init__(self) -> None:
        pass

    def __repr__(self) -> str:
        return f"Custom Element: {self.name}"

    @abstractmethod
    def generate_profile(self):
        # TODO: Doc here that this can be a list if multiple profiles are generated
        raise NotImplementedError("Custom elements must be modified in some way.")

    @staticmethod
    @abstractmethod
    def __keys__() -> dict:
        raise NotImplementedError(
            "Custom elements must have a list of keys to generate a profile."
        )
