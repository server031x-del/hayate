"""Domain errors with messages suitable for the CLI."""


class HayateError(Exception):
    """Base class for expected HAYATE failures."""


class ModelRegistryError(HayateError):
    """The model registry is missing or malformed."""


class SafetensorsInspectionError(HayateError):
    """A safetensors header is invalid, corrupt, or unsafe to parse."""


class LoaderValidationError(HayateError):
    """A model does not meet a loader's structural contract."""


class LoaderNotReadyError(HayateError):
    """A recognized format has no enabled execution implementation yet."""


class UpstreamContractError(HayateError):
    """A maybleMyers/h3 checkout does not match the required engine contract."""


class GenerationPreflightError(HayateError):
    """A generation request cannot safely enter the upstream engine."""
