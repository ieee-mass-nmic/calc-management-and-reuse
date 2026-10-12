"""Backend contract. A real PCU adapter must declare and validate its own dependencies."""

from typing import Any, Protocol

from ..graph import ArtifactSpec


class Backend(Protocol):
    @property
    def context(self) -> dict:
        """Backend identity, code digest and numerical environment."""
        ...

    def execute(self, spec: ArtifactSpec, parents: list[Any]) -> Any:
        """Pure deterministic operator under the declared execution protocol."""
        ...
