import os
import platform


def _find_font(*candidates: str) -> str:
    """Return the first font path that exists, or the last candidate as fallback."""
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[-1]


# ── Platform-aware font resolution ──────────────────────────────────
_IS_WIN = platform.system() == "Windows"
_IS_MAC = platform.system() == "Darwin"

if _IS_WIN:
    _SERIF = _find_font(
        r"C:\Windows\Fonts\times.ttf",
        r"C:\Windows\Fonts\georgia.ttf",
        r"C:\Windows\Fonts\arial.ttf",
    )
    _BLACKLETTER = _find_font(
        r"C:\Windows\Fonts\OLDENGL.TTF",
        r"C:\Windows\Fonts\times.ttf",
    )
elif _IS_MAC:
    _SERIF = _find_font(
        "/Library/Fonts/Times New Roman.ttf",
        "/System/Library/Fonts/Times.ttc",
    )
    _BLACKLETTER = _find_font(
        "/Library/Fonts/Canterbury.ttf",
        _SERIF,
    )
else:  # Linux / other
    _SERIF = _find_font(
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    )
    _BLACKLETTER = _find_font(
        "/usr/share/fonts/truetype/unifrakturmaguntia/UnifrakturMaguntia.ttf",
        _SERIF,
    )


def _lang(tesseract_lang: str, *, blackletter: str | None = None) -> dict:
    return {
        "tesseract_lang": tesseract_lang,
        "default_font": _SERIF,
        "blackletter_font": blackletter or _BLACKLETTER,
    }


# ── Language configurations ─────────────────────────────────────────
# Latin-script languages
LANGUAGE_CONFIGS = {
    "eng": _lang("eng"),
    "deu": _lang("deu"),
    "fra": _lang("fra"),
    "spa": _lang("spa"),
    "ltz": _lang("ltz"),
    # Non-Latin scripts (blackletter = serif fallback)
    "rus": _lang("rus", blackletter=_SERIF),
    "ell": _lang("ell", blackletter=_SERIF),
    "ara": _lang("ara", blackletter=_SERIF),
    "heb": _lang("heb", blackletter=_SERIF),
    "kat": _lang("kat", blackletter=_SERIF),
}
