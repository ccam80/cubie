import attrs

@attrs.define
class IntegratorAlgorithmData:
    """
    Data class to store and derive information from integrator loop settings. Handles the data I/O for the integrator
    algorithm, and invalidates cached loop functions when contained information changes.
    """
    compile_settings: dict = attrs.field(validator=attrs.validators.instance_of(dict), default={})
    precision: float = attrs.field(validator=attrs.validators.instance_of(float), default=float)
    num_drivers: int = attrs.field(validator=attrs.validators.instance_of(int), default=1)

    @property
    def sizes(self):
        """Returns a dictionary of sizes for the integrator algorithm data."""
        return {
            'n_drivers': self.num_drivers
        }