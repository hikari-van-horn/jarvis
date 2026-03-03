from typing import Type, Any
from google.protobuf.message import Message
from google.protobuf import json_format
from google.protobuf.descriptor import FieldDescriptor, Descriptor
import logging


# ---------------------------------------------------------------------------
# Proto field type → JSON Schema type mapping
# ---------------------------------------------------------------------------
_SCALAR_TYPE_MAP: dict[int, dict] = {
    FieldDescriptor.TYPE_STRING:   {"type": "string"},
    FieldDescriptor.TYPE_BYTES:    {"type": "string", "contentEncoding": "base64"},
    FieldDescriptor.TYPE_BOOL:     {"type": "boolean"},
    FieldDescriptor.TYPE_FLOAT:    {"type": "number"},
    FieldDescriptor.TYPE_DOUBLE:   {"type": "number"},
    FieldDescriptor.TYPE_INT32:    {"type": "integer"},
    FieldDescriptor.TYPE_INT64:    {"type": "integer"},
    FieldDescriptor.TYPE_UINT32:   {"type": "integer", "minimum": 0},
    FieldDescriptor.TYPE_UINT64:   {"type": "integer", "minimum": 0},
    FieldDescriptor.TYPE_SINT32:   {"type": "integer"},
    FieldDescriptor.TYPE_SINT64:   {"type": "integer"},
    FieldDescriptor.TYPE_FIXED32:  {"type": "integer", "minimum": 0},
    FieldDescriptor.TYPE_FIXED64:  {"type": "integer", "minimum": 0},
    FieldDescriptor.TYPE_SFIXED32: {"type": "integer"},
    FieldDescriptor.TYPE_SFIXED64: {"type": "integer"},
}

# Well-known proto types that map directly to JSON Schema primitives
_WELL_KNOWN_TYPES: dict[str, dict] = {
    "google.protobuf.Timestamp":  {"type": "string", "format": "date-time"},
    "google.protobuf.Duration":   {"type": "string", "format": "duration"},
    "google.protobuf.StringValue":{"type": "string"},
    "google.protobuf.BoolValue":  {"type": "boolean"},
    "google.protobuf.Int32Value":  {"type": "integer"},
    "google.protobuf.Int64Value":  {"type": "integer"},
    "google.protobuf.FloatValue":  {"type": "number"},
    "google.protobuf.DoubleValue": {"type": "number"},
}


def _descriptor_to_schema(descriptor: Descriptor, visited: set[str] | None = None) -> dict:
    """Recursively convert a protobuf Descriptor to a JSON Schema object."""
    if visited is None:
        visited = set()

    full_name = descriptor.full_name

    # Guard against circular references
    if full_name in visited:
        return {"type": "object", "description": f"(circular ref: {full_name})"}
    visited = visited | {full_name}  # immutable copy for each branch

    # Well-known google types → direct mapping
    if full_name in _WELL_KNOWN_TYPES:
        return dict(_WELL_KNOWN_TYPES[full_name])

    properties: dict[str, dict] = {}

    for field in descriptor.fields:
        field_schema = _field_to_schema(field, visited)
        # Use snake_case name (proto field name) to match preserving_proto_field_name=True
        properties[field.name] = field_schema

    schema: dict = {"type": "object", "properties": properties}
    if descriptor.GetOptions().HasField("map_entry") if False else False:
        pass  # handled in _field_to_schema for map fields
    return schema


def _field_to_schema(field: FieldDescriptor, visited: set[str]) -> dict:
    """Convert a single proto field descriptor to its JSON Schema representation."""

    # ------------------------------------------------------------------ map<k,v>
    # Map fields appear as repeated message with map_entry option set.
    if (field.message_type is not None
            and field.message_type.GetOptions().map_entry):
        value_field = field.message_type.fields_by_name["value"]
        value_schema = _field_to_schema(value_field, visited)
        return {
            "type": "object",
            "additionalProperties": value_schema,
        }

    # ------------------------------------------------------------------ enum
    if field.type == FieldDescriptor.TYPE_ENUM:
        enum_values = [v.name for v in field.enum_type.values]
        base: dict = {"type": "string", "enum": enum_values}
        return _wrap_repeated(field, base)

    # ------------------------------------------------------------------ nested message
    if field.type == FieldDescriptor.TYPE_MESSAGE:
        nested = _descriptor_to_schema(field.message_type, visited)
        return _wrap_repeated(field, nested)

    # ------------------------------------------------------------------ scalar
    base = dict(_SCALAR_TYPE_MAP.get(field.type, {"type": "string"}))
    return _wrap_repeated(field, base)


def _wrap_repeated(field: FieldDescriptor, item_schema: dict) -> dict:
    """Wrap item_schema in an array schema if the field is repeated."""
    if field.is_repeated:
        return {"type": "array", "items": item_schema}
    return item_schema


class ProtobufJsonConverter:
    """Protocol Buffer 与 JSON 相互转换工具类，以及 JSON Schema 生成。"""

    @staticmethod
    def proto_to_json(
        proto_msg: Message,
        indent: int = 2,
        preserving_proto_field_name: bool = True,
        including_default_value_fields: bool = False
    ) -> str:
        """将Protobuf消息转换为JSON字符串"""
        try:
            kwargs = {
                'indent': indent,
                'preserving_proto_field_name': preserving_proto_field_name
            }
            return json_format.MessageToJson(proto_msg, **kwargs)
        except json_format.Error as e:
            logging.error(f"Protobuf to JSON conversion failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during conversion: {e}")
            raise

    @staticmethod
    def json_to_proto(
        json_str: str,
        proto_class: Type[Message],
        ignore_unknown_fields: bool = False
    ) -> Message:
        """将JSON字符串转换为Protobuf消息"""
        try:
            proto_msg = proto_class()
            json_format.Parse(
                json_str,
                proto_msg,
                ignore_unknown_fields=ignore_unknown_fields
            )
            return proto_msg
        except json_format.ParseError as e:
            logging.error(f"JSON to Protobuf parsing failed: {e}")
            raise
        except Exception as e:
            logging.error(f"Unexpected error during parsing: {e}")
            raise

    @staticmethod
    def proto_to_json_schema(
        proto_class: Type[Message],
        title: str | None = None,
        description: str | None = None,
    ) -> dict:
        """Generate a JSON Schema (draft-07) dict from a Protobuf message class.

        The schema uses snake_case field names (matching ``preserving_proto_field_name=True``).
        It can be passed directly to an LLM as a structured-output schema
        (e.g. via ``ChatOpenAI.with_structured_output(schema)``).

        Args:
            proto_class: The generated protobuf message class (e.g. ``persona_pb2.UserPersona``).
            title:       Optional ``$schema`` title; defaults to the message's simple name.
            description: Optional description string added to the root schema.

        Returns:
            A JSON-serialisable dict representing the JSON Schema.

        Example::

            schema = ProtobufJsonConverter.proto_to_json_schema(UserPersona)
            llm.with_structured_output(schema)
        """
        descriptor: Descriptor = proto_class.DESCRIPTOR
        schema = _descriptor_to_schema(descriptor)
        schema["$schema"] = "http://json-schema.org/draft-07/schema#"
        schema["title"] = title or descriptor.name
        if description:
            schema["description"] = description
        return schema