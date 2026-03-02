from typing import Type, Any
from google.protobuf.message import Message
from google.protobuf import json_format
import logging

class ProtobufJsonConverter:
    """Protocol Buffer与JSON相互转换工具类"""
    
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
            # import inspect
            # sig = inspect.signature(json_format.MessageToJson)
            # if 'always_print_fields_with_no_presence' in sig.parameters:
            #     kwargs['always_print_fields_with_no_presence'] = including_default_value_fields
            # elif 'including_default_value_fields' in sig.parameters:
            #     kwargs['including_default_value_fields'] = including_default_value_fields
                
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