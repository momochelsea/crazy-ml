def get_sec(e, s) -> float:
    return int((e - s) * 1000) / 1000.0


def get_ms(e, s) -> int:
    return int((e - s) * 1000)


def get_percent(p) -> float:
    return int(p * 100 * 100) / 100.0
