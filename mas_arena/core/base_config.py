from ..core.component import SerializableComponent


class BaseConfig(SerializableComponent):
    """
    Base configuration class for the MAS Arena framework.

    This class serves as a base for all configuration classes in the MAS Arena framework.
    It inherits from SerializableComponent to ensure that configurations can be serialized
    and deserialized as needed.
    """

    def save(self, path: str, **kwargs):
        return super().save_component(path, **kwargs)
