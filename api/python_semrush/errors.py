from __future__ import absolute_import, print_function, unicode_literals


class BaseSemrushError(Exception):
    pass


class SemRushKeyError(BaseSemrushError):
    pass


class SemRushRegionalDatabaseError(BaseSemrushError):
    pass