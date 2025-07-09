from pydantic._internal._model_construction import ModelMetaclass
from pydantic.v1 import BaseModel
from .registry import register_model, COMPONENT_REGISTRY
from .utils import custom_serializer
import json
from typing import List, Dict, Any


class ComponentRegistry(ModelMetaclass):

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Custom metaclass to register components in a global registry.

        This metaclass automatically registers any class that inherits from
        SerializableComponent into the global component registry.
        """
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        register_model(name, cls)
        return cls


class SerializableComponent(BaseModel, metaclass=ComponentRegistry):
    class_name: str = None
    model_config = {"arbitrary_types_allowed": True, "extra": "allow", "protected_namespaces": (),
                    "validate_assignment": False}

    def save_component(self, ):

        def to_json(self, use_indent: bool = False, ignore: List[str] = [], **kwargs) -> str:
            """
            Convert the BaseModule to a JSON string.

            Args:
                use_indent: Whether to use indentation
                ignore: List of field names to ignore
                **kwargs (Any): Additional keyword arguments

            Returns:
                str: The JSON string
            """
            if use_indent:
                kwargs["indent"] = kwargs.get("indent", 4)
            else:
                kwargs.pop("indent", None)
            if kwargs.get("default", None) is None:
                kwargs["default"] = custom_serializer
            data = self.to_dict(exclude_none=True)
            for ignore_field in ignore:
                data.pop(ignore_field, None)
            return json.dumps(data, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any], **kwargs):
        """
        Create an instance of the class from a dictionary.

        Args:
            data (Dict[str, Any]): The dictionary to create the instance from.
            **kwargs: Additional keyword arguments.

        Returns:
            SerializableComponent: An instance of the class.
        """
        use_logger = kwargs.get("log", True)
        try:
            class_name = data.get("class_name", None)
            if class_name:
                cls = COMPONENT_REGISTRY.get_component(class_name)
            component = cls._create_instance(data)
        finally:
            pass

        return component

    def to_dict(self, exclude_none: bool = True, ignore: List[str] = [], **kwargs) -> dict:
        """
        Convert the BaseModule to a dictionary.

        Args:
            exclude_none: Whether to exclude fields with None values
            ignore: List of field names to ignore
            **kwargs (Any): Additional keyword arguments

        Returns:
            dict: Dictionary containing the object data
        """
        data = {}
        for field_name, _ in type(self).model_fields.items():
            if field_name in ignore:
                continue
            field_value = getattr(self, field_name, None)
            if exclude_none and field_value is None:
                continue
            if isinstance(field_value, SerializableComponent):
                data[field_name] = field_value.to_dict(exclude_none=exclude_none, ignore=ignore)
            elif isinstance(field_value, list):
                data[field_name] = [
                    item.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(item,
                                                                                         SerializableComponent) else item
                    for item in field_value
                ]
            elif isinstance(field_value, dict):
                data[field_name] = {
                    key: value.to_dict(exclude_none=exclude_none, ignore=ignore) if isinstance(value,
                                                                                               SerializableComponent) else value
                    for key, value in field_value.items()
                }
            else:
                data[field_name] = field_value

        return data
