# -*- coding: utf-8 -*-
"""Wrapper around the SEMrush API."""
# :copyright: (c) 2015 Jeremy Storer and individual contributors,
#                 All rights reserved.
# :license:   MIT License, see LICENSE for more details.


from collections import namedtuple
from .semrush import SEMRushClient

version_info_t = namedtuple(
    'version_info_t', ('major', 'minor', 'micro', 'releaselevel', 'serial'),
)

VERSION = version_info_t(0, 1, 2, '', '')
__version__ = '{0.major}.{0.minor}.{0.micro}{0.releaselevel}'.format(VERSION)
__author__ = 'Jeremy Storer'
__contact__ = 'storerjeremy@gmail.com'
__homepage__ = 'http://github.com/storerjeremy/python-semrush'
__docformat__ = 'restructuredtext'

# -eof meta-
