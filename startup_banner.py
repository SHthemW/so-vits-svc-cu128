import os

_K = 73
_EMITTED_ENVIRONMENT_VARIABLE = "SOVITS_STARTUP_BANNER_EMITTED"
_PAYLOAD = (
    (106, 105, 26, 38, 100, 31, 0, 29, 26, 100, 26, 31, 10, 105, 125, 103, 120, 105, 62, 32, 61, 33, 105, 10, 28, 13, 8, 105, 120, 123, 103, 113),
    (100, 105, 0, 39, 61, 44, 46, 59, 40, 61, 44, 45, 105, 43, 48, 105, 26, 1, 30, 105, 102, 105, 26, 1, 61, 33, 44, 36, 30, 9, 14, 32, 61, 33, 60, 43),
    (100, 105, 26, 38, 100, 31, 0, 29, 26, 100, 26, 31, 10, 105, 32, 58, 105, 38, 57, 44, 39, 100, 58, 38, 60, 59, 42, 44, 105, 58, 38, 47, 61, 62, 40, 59, 44, 103, 105, 29, 33, 32, 58, 105, 43, 60, 39, 45, 37, 44, 45, 105, 43, 60, 32, 37, 45, 105, 32, 58, 105, 47, 59, 44, 44, 105, 40, 39, 45, 105, 39, 38, 61, 105, 47, 38, 59, 105, 42, 38, 36, 36, 44, 59, 42, 32, 40, 37, 105, 60, 58, 44, 103),
)


def _decode(values):
    return bytes(value ^ _K for value in values).decode("utf-8")


def emit_startup_banner(title=None):
    print()
    for line in _PAYLOAD:
        print(_decode(line))
    if title:
        print(title)
    print()
    os.environ[_EMITTED_ENVIRONMENT_VARIABLE] = "1"


def emit_startup_banner_once(title=None):
    if os.environ.get(_EMITTED_ENVIRONMENT_VARIABLE) == "1":
        return
    emit_startup_banner(title)
