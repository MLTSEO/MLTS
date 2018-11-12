from __future__ import absolute_import, print_function, unicode_literals


class BaseGscError(Exception):
    pass


class GscConfigError(BaseGscError):
    pass


class GscApiError(BaseGscError):
    pass
