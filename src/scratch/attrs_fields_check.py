import attrs

@attrs.define
class CheckerClass:
    field1: str = attrs.field(validator=attrs.validators.instance_of(str), default="This one exists")
    field2: int = attrs.field(validator=attrs.validators.instance_of(int), default=42)
    field3: float = attrs.field(validator=attrs.validators.instance_of(float), default=3.14)

checker = CheckerClass()
attrs.fields(checker.__class__)
'field1' in [field.name for field in attrs.fields(checker.__class__)]