import re
from typing import Any
from datetime import datetime, date


def custom_serializer(obj: Any):
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode()
    if isinstance(obj, (datetime, date)):
        return obj.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "read") and hasattr(obj, "name"):
        return f"<FileObject name={getattr(obj, 'name', 'unknown')}>"
    if callable(obj):
        return obj.__name__
    if hasattr(obj, "__class__"):
        return obj.__repr__() if hasattr(obj, "__repr__") else obj.__class__.__name__

    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def generate_dynamic_class_name(base_name: str) -> str:
    base_name = base_name.strip()

    cleaned_name = re.sub(r'[^a-zA-Z0-9\s]', ' ', base_name)
    components = cleaned_name.split()
    class_name = ''.join(x.capitalize() for x in components)

    return class_name if class_name else 'DefaultClassName'
