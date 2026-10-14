"""Browser-compatible length budgets; never mutate or truncate request values."""

import json
import math


def utf16_length(text: str) -> int:
    """Match JavaScript string.length and HTML maxlength for valid Unicode."""
    try:
        return len(text.encode("utf-16-le")) // 2
    except UnicodeEncodeError:
        raise ValueError(
            "Text must contain valid Unicode, without unpaired surrogates."
        )


def _float_json_length(value: float) -> int:
    """Measure finite JSON numbers with JSON.stringify's decimal/exponent ranges.

    Python repr adds .0 and exponent zero-padding and switches notation earlier.
    Normalize those formatting differences without changing the native value.
    """
    if not math.isfinite(value):
        raise ValueError("JSON numbers must be finite.")
    if value == 0:
        return 1
    sign = int(value < 0)
    mantissa, _, exponent = repr(abs(value)).partition("e")
    whole, _, fraction = mantissa.partition(".")
    digits = whole + fraction
    leading = len(digits) - len(digits.lstrip("0"))
    position = len(whole) + int(exponent or 0) - leading
    digits = digits.lstrip("0").rstrip("0")
    power = position - 1
    if -6 <= power < 21:
        if position <= 0:
            return sign + 2 - position + len(digits)
        if position >= len(digits):
            return sign + position
        return sign + len(digits) + 1
    # Mantissa, optional decimal point, e, exponent sign, exponent digits.
    return sign + len(digits) + int(len(digits) > 1) + 2 + len(str(abs(power)))


def compact_json_length(value) -> int:
    """Count compact JSON UTF-16 units without serialization or recursive frames."""
    size = 0
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            utf16_length(item)
            size += utf16_length(json.dumps(item, ensure_ascii=False))
        elif item is None:
            size += 4
        elif isinstance(item, bool):
            size += 4 if item else 5
        elif isinstance(item, int):
            size += len(str(item))
        elif isinstance(item, float):
            size += _float_json_length(item)
        elif isinstance(item, list):
            size += 2 + max(0, len(item) - 1)
            pending.extend(item)
        elif isinstance(item, dict):
            if any(not isinstance(key, str) for key in item):
                raise ValueError("JSON object keys must be strings.")
            size += 2 + max(0, len(item) - 1) + len(item)
            pending.extend(item.keys())
            pending.extend(item.values())
        else:
            raise ValueError("State must contain JSON-compatible values.")
    return size
