"""Release-freeze manifests required for prospective candidate screening."""

from .freeze import FreezeManifestError, write_freeze_manifest

__all__ = ["FreezeManifestError", "write_freeze_manifest"]
