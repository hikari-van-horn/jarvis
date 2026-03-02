import pytest
import json
from google.protobuf import json_format

from src.utils.proto2json import ProtobufJsonConverter
from src.memory import user_context_schema_pb2

class TestProtobufJsonConverter:
    
    def test_proto_to_json(self):
        """测试 Protobuf 转 JSON"""
        # 创建一个简单的 Protobuf message
        msg = user_context_schema_pb2.UserPersona(user_id="test_user_1")
        msg.demographics.preferred_name = "Jarvis"
        
        # 转换为 JSON 返回字符串
        json_str = ProtobufJsonConverter.proto_to_json(
            msg, 
            indent=None,
            preserving_proto_field_name=True
        )
        
        # 验证返回的内容
        parsed_dict = json.loads(json_str)
        assert parsed_dict.get("user_id") == "test_user_1"
        assert parsed_dict.get("demographics", {}).get("preferred_name") == "Jarvis"

    def test_json_to_proto(self):
        """测试 JSON 转 Protobuf"""
        json_str = '{"user_id": "test_user_2", "demographics": {"preferred_name": "Ultron"}}'
        
        # 转换回 Protobuf message
        msg = ProtobufJsonConverter.json_to_proto(
            json_str, 
            user_context_schema_pb2.UserPersona
        )
        
        # 验证转换类型及内容
        assert isinstance(msg, user_context_schema_pb2.UserPersona)
        assert msg.user_id == "test_user_2"
        assert msg.demographics.preferred_name == "Ultron"

    def test_json_to_proto_with_unknown_fields(self):
        """测试包含未知字段时的反序列化行为（不忽略）"""
        json_str = '{"user_id": "test", "extra_field": 123}'
        
        # 默认情况下遇到未知属性会抛出 ParseError
        with pytest.raises(json_format.ParseError):
            ProtobufJsonConverter.json_to_proto(
                json_str, 
                user_context_schema_pb2.UserPersona,
                ignore_unknown_fields=False
            )

    def test_json_to_proto_ignore_unknown_fields(self):
        """测试包含未知字段时的反序列化行为（忽略）"""
        json_str = '{"user_id": "test", "extra_field": 123}'
        
        # 配置忽略未知属性后应该成功解析
        msg = ProtobufJsonConverter.json_to_proto(
            json_str, 
            user_context_schema_pb2.UserPersona,
            ignore_unknown_fields=True
        )
        
        assert msg.user_id == "test"
