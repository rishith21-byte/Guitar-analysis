import math

STRINGS = {
    "E2": 82.41,
    "A2": 110.00,
    "D3": 146.83,
    "G3": 196.00,
    "B3": 246.94,
    "E4": 329.63,
}


def frequency_to_note(frequency):
    if frequency is None or frequency <= 0:
        return "--"
    return min(STRINGS, key=lambda note: abs(STRINGS[note] - frequency))


def get_closest_string(frequency):
    if frequency is None or frequency <= 0:
        return None
    return min(STRINGS, key=lambda note: abs(STRINGS[note] - frequency))


def calculate_deviation(frequency, note):
    if frequency is None or frequency <= 0 or note not in STRINGS:
        return 0.0
    return frequency - STRINGS[note]


def tuning_status(deviation):
    if deviation is None:
        return "UNKNOWN"
    if deviation < -5:
        return "FLAT"
    if deviation > 5:
        return "SHARP"
    return "IN TUNE"


def frequency_to_cents(frequency, target):
    if frequency is None or target is None or frequency <= 0 or target <= 0:
        return 0.0
    return 1200 * math.log2(frequency / target)
