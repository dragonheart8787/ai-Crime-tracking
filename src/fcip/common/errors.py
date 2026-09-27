"""Project-wide exception types. Invariant violations always raise; nothing is silently repaired."""


class FcipError(Exception):
    """Base class for all project errors."""


class ConfigError(FcipError):
    """Invalid or inconsistent configuration."""


class SchemaError(FcipError):
    """A table does not conform to its registered schema."""


class InvariantViolation(FcipError):
    """A generated dataset violates a documented invariant."""


class FutureAccessError(FcipError):
    """An attempt to read data after an as-of cutoff (used from Milestone 1, checkpoint 2)."""
