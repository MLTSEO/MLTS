from __future__ import absolute_import, print_function, unicode_literals


class BaseGaError(Exception):
    pass


class GaConfigError(BaseGaError):
    pass


class GaApiError(BaseGaError):
    pass
