import json

import pytest
from google.protobuf import json_format

from src.agent.memory import persona_pb2
from src.utils.proto2json import ProtobufJsonConverter


class TestProtobufJsonConverter:
    def test_proto_to_json(self):
        """测试 Protobuf 转 JSON"""
        # 创建一个简单的 Protobuf message
        msg = persona_pb2.UserPersona(user_id="test_user_1")
        msg.demographics.preferred_name = "Jarvis"

        # 转换为 JSON 返回字符串
        json_str = ProtobufJsonConverter.proto_to_json(msg, indent=None, preserving_proto_field_name=True)

        # 验证返回的内容
        parsed_dict = json.loads(json_str)
        assert parsed_dict.get("user_id") == "test_user_1"
        assert parsed_dict.get("demographics", {}).get("preferred_name") == "Jarvis"

    def test_json_to_proto(self):
        """测试 JSON 转 Protobuf"""
        json_str = '{"user_id": "test_user_2", "demographics": {"preferred_name": "Ultron"}}'

        # 转换回 Protobuf message
        msg = ProtobufJsonConverter.json_to_proto(json_str, persona_pb2.UserPersona)

        # 验证转换类型及内容
        assert isinstance(msg, persona_pb2.UserPersona)
        assert msg.user_id == "test_user_2"
        assert msg.demographics.preferred_name == "Ultron"

    def test_json_to_proto_with_unknown_fields(self):
        """测试包含未知字段时的反序列化行为（不忽略）"""
        json_str = '{"user_id": "test", "extra_field": 123}'

        # 默认情况下遇到未知属性会抛出 ParseError
        with pytest.raises(json_format.ParseError):
            ProtobufJsonConverter.json_to_proto(json_str, persona_pb2.UserPersona, ignore_unknown_fields=False)

    def test_json_to_proto_ignore_unknown_fields(self):
        """测试包含未知字段时的反序列化行为（忽略）"""
        json_str = '{"user_id": "test", "extra_field": 123}'

        # 配置忽略未知属性后应该成功解析
        msg = ProtobufJsonConverter.json_to_proto(json_str, persona_pb2.UserPersona, ignore_unknown_fields=True)

        assert msg.user_id == "test"


class TestProtoToJsonSchema:
    @pytest.fixture(scope="class")
    def schema(self):
        return ProtobufJsonConverter.proto_to_json_schema(persona_pb2.UserPersona)

    # ------------------------------------------------------------------
    # Root schema structure
    # ------------------------------------------------------------------

    def test_schema_has_correct_keys(self, schema):
        assert schema["$schema"] == "http://json-schema.org/draft-07/schema#"
        assert schema["title"] == "UserPersona"
        assert schema["type"] == "object"
        assert "properties" in schema

    def test_custom_title_and_description(self):
        schema = ProtobufJsonConverter.proto_to_json_schema(
            persona_pb2.UserPersona,
            title="MyPersona",
            description="A user memory document",
        )
        assert schema["title"] == "MyPersona"
        assert schema["description"] == "A user memory document"

    def test_top_level_fields_present(self, schema):
        props = schema["properties"]
        assert "user_id" in props
        assert "demographics" in props
        assert "preferences" in props
        assert "work_context" in props
        assert "financial_profile" in props

    # ------------------------------------------------------------------
    # Scalar types
    # ------------------------------------------------------------------

    def test_string_field_type(self, schema):
        assert schema["properties"]["user_id"] == {"type": "string"}

    def test_nested_string_fields(self, schema):
        demo_props = schema["properties"]["demographics"]["properties"]
        assert demo_props["preferred_name"] == {"type": "string"}

    # ------------------------------------------------------------------
    # Nested messages
    # ------------------------------------------------------------------

    def test_nested_message_is_object(self, schema):
        assert schema["properties"]["demographics"]["type"] == "object"

    def test_deeply_nested_location(self, schema):
        demo_props = schema["properties"]["demographics"]["properties"]
        assert "home_location" in demo_props
        loc = demo_props["home_location"]
        assert loc["type"] == "object"
        assert loc["properties"]["city"] == {"type": "string"}
        assert loc["properties"]["country_code"] == {"type": "string"}
        assert loc["properties"]["timezone"] == {"type": "string"}

    # ------------------------------------------------------------------
    # Repeated fields → array
    # ------------------------------------------------------------------

    def test_repeated_field_is_array(self, schema):
        demo_props = schema["properties"]["demographics"]["properties"]
        edu_hist = demo_props["education_history"]
        assert edu_hist["type"] == "array"
        assert edu_hist["items"]["type"] == "object"

    def test_repeated_string_field_is_array_of_strings(self, schema):
        pref_props = schema["properties"]["preferences"]["properties"]
        langs = pref_props["languages"]
        assert langs["type"] == "array"
        assert langs["items"] == {"type": "string"}

    # ------------------------------------------------------------------
    # Enum fields
    # ------------------------------------------------------------------

    def test_enum_field_type(self, schema):
        demo_props = schema["properties"]["demographics"]["properties"]
        edu_items = demo_props["education_history"]["items"]
        degree = edu_items["properties"]["degree"]
        assert degree["type"] == "string"
        assert set(degree["enum"]) == {"DEGREE_UNSPECIFIED", "BACHELOR", "MASTER", "PHD", "POSTDOC"}

    def test_fact_metadata_enum(self, schema):
        meta = schema["properties"]["demographics"]["properties"]["meta"]
        source = meta["properties"]["source"]
        assert source["type"] == "string"
        assert "EXPLICIT_USER_STATEMENT" in source["enum"]
        assert "IMPLICIT_INFERENCE" in source["enum"]
        assert "THIRD_PARTY_EXTENSIONS" in source["enum"]

    # ------------------------------------------------------------------
    # Map fields → additionalProperties
    # ------------------------------------------------------------------

    def test_map_field_is_object_with_additional_properties(self, schema):
        pref_props = schema["properties"]["preferences"]["properties"]
        coding_prefs = pref_props["coding_preferences"]
        assert coding_prefs["type"] == "object"
        assert coding_prefs["additionalProperties"] == {"type": "string"}

    # ------------------------------------------------------------------
    # Well-known type: google.protobuf.Timestamp
    # ------------------------------------------------------------------

    def test_timestamp_field_is_date_time_string(self, schema):
        meta = schema["properties"]["demographics"]["properties"]["meta"]
        created_at = meta["properties"]["created_at"]
        assert created_at == {"type": "string", "format": "date-time"}
        last_verified = meta["properties"]["last_verified_at"]
        assert last_verified == {"type": "string", "format": "date-time"}

    # ------------------------------------------------------------------
    # Numeric types
    # ------------------------------------------------------------------

    def test_float_field_is_number(self, schema):
        meta = schema["properties"]["demographics"]["properties"]["meta"]
        confidence = meta["properties"]["confidence_score"]
        assert confidence == {"type": "number"}

    # ------------------------------------------------------------------
    # Schema is JSON-serialisable
    # ------------------------------------------------------------------

    def test_schema_is_json_serialisable(self, schema):
        serialised = json.dumps(schema)
        roundtrip = json.loads(serialised)
        assert roundtrip["title"] == "UserPersona"

    # ------------------------------------------------------------------
    # Standalone message classes
    # ------------------------------------------------------------------

    def test_location_schema_standalone(self):
        schema = ProtobufJsonConverter.proto_to_json_schema(persona_pb2.Location)
        assert schema["title"] == "Location"
        assert schema["properties"]["city"] == {"type": "string"}
        assert schema["properties"]["country_code"] == {"type": "string"}

    def test_education_schema_standalone(self):
        schema = ProtobufJsonConverter.proto_to_json_schema(persona_pb2.Education)
        props = schema["properties"]
        assert props["major"] == {"type": "string"}
        assert props["institution"] == {"type": "string"}
        assert props["start_year"] == {"type": "integer"}
        degree = props["degree"]
        assert degree["type"] == "string"
        assert "PHD" in degree["enum"]
