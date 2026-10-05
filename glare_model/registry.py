"""Name -> builder registries so configs can build models and data sources by string.

A config names a registered class or factory function (`MODEL.NAME: GlareRemovalNAFNet`) and the
registry calls it with keyword arguments taken from the config section (UPPERCASE keys lowered).
Registration is the `@REGISTRY.register` decorator, which runs when the defining module is
imported; `glare_model/training/component_builders.py` imports every registering module.
"""

from collections.abc import Callable
from typing import Any


class NamedRegistry:
    """A mapping from name to a class or factory function, with decorator registration."""

    def __init__(self, registry_name: str):
        self.registry_name = registry_name
        self.registered_builders: dict[str, Callable[..., Any]] = {}

    def register(self, builder: Callable[..., Any]) -> Callable[..., Any]:
        """Decorator: register `builder` under its own `__name__` (duplicates are an error)."""
        builder_name = builder.__name__
        if builder_name in self.registered_builders:
            raise KeyError(f"{builder_name} is already registered in the {self.registry_name} registry")
        self.registered_builders[builder_name] = builder
        return builder

    def get(self, builder_name: str) -> Callable[..., Any]:
        """Return the registered builder, failing loudly with the list of known names."""
        if builder_name not in self.registered_builders:
            known_names = ", ".join(sorted(self.registered_builders))
            raise KeyError(f"unknown {self.registry_name} {builder_name!r}; known: {known_names}")
        return self.registered_builders[builder_name]

    def build(self, builder_name: str, **builder_kwargs: Any) -> Any:
        """Call the registered builder `builder_name` with `builder_kwargs`."""
        return self.get(builder_name)(**builder_kwargs)

    def names(self) -> list[str]:
        """Return all registered names, sorted."""
        return sorted(self.registered_builders)


def build_from_config_section(registry: NamedRegistry, config_section: dict[str, Any], **extra_kwargs: Any) -> Any:
    """Build `config_section["NAME"]` with the section's other keys lowered into kwargs.

    `{"NAME": "GlareRemovalNAFNet", "BASE_WIDTH": 16}` -> `GlareRemovalNAFNet(base_width=16)`.
    """
    builder_kwargs = {key.lower(): value for key, value in config_section.items() if key != "NAME"}
    return registry.build(config_section["NAME"], **builder_kwargs, **extra_kwargs)


MODEL_REGISTRY = NamedRegistry("model")
SOURCE_FACE_PROVIDER_REGISTRY = NamedRegistry("source face provider")
GLARE_SYNTHESIZER_REGISTRY = NamedRegistry("glare synthesizer factory")
