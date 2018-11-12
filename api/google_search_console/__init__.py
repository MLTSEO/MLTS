# -*- coding: utf-8 -*-
"""Wrapper around the GSC API."""


from collections import namedtuple
from .gsc import GscClient

version_info_t = namedtuple(
    'version_info_t', ('major', 'minor', 'micro', 'releaselevel', 'serial'),
)



VERSION = version_info_t(0, 1, 2, '', '')
__version__ = '{0.major}.{0.minor}.{0.micro}{0.releaselevel}'.format(VERSION)
__author__ = 'JR Oakes'
__contact__ = 'jroakes@gmail.com'
__homepage__ = 'http://github.com/jroakes/'
__docformat__ = 'restructuredtext'

# -eof meta-
