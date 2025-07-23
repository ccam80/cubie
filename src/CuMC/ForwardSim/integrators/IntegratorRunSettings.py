import attrs

@attrs.define
class IntegatorRunSettings:
    """Container for runtime/timing settings that are commonly grouped together."""
    duration: float = attrs.field(default=1.0, validator=attrs.validators.instance_of(float))
    warmup: float = attrs.field(default=0.0, validator=attrs.validators.instance_of(float))
    dt_min: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))
    dt_max: float = attrs.field(default=1.0, validator=attrs.validators.instance_of(float))
    dt_save: float = attrs.field(default=0.1, validator=attrs.validators.instance_of(float))
    dt_summarise: float = attrs.field(default=0.1, validator=attrs.validators.instance_of(float))
    atol: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))
    rtol: float = attrs.field(default=1e-6, validator=attrs.validators.instance_of(float))

    def __attrs_post_init__(self):
        """Validate timing relationships."""
        if self.dt_min > self.dt_max:
            raise ValueError("dt_min must be <= dt_max")
        if self.dt_save < self.dt_min:
            raise ValueError("dt_save must be >= dt_min")
        if self.dt_summarise < self.dt_save:
            raise ValueError("dt_summarise must be >= dt_save")