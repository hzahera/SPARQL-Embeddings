from __future__ import annotations

from collections.abc import Callable


def identity(text: str) -> str:
    return text


_TRANSFORMS: dict[str, Callable[[str], str]] = {
    "identity": identity,
}


def register_transform(name: str, fn: Callable[[str], str]) -> None:
    if not name:
        raise ValueError("Transform name must be non-empty")
    _TRANSFORMS[name] = fn


def get_transform(name: str) -> Callable[[str], str]:
    try:
        return _TRANSFORMS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown input transform {name!r}. Registered: {sorted(_TRANSFORMS)}"
        ) from exc
