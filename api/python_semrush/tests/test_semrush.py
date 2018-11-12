# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals
import os
from unittest import TestCase
try:
    from unittest.mock import patch
except:
    from mock import patch
from python_semrush.semrush import SemrushClient
from requests import Response


def semrush_response_bytes(filename):
    with open(os.path.join(os.path.dirname(__file__), filename), 'rb') as f:
        return f.read()


class SemrushTestCase(TestCase):

    def test_parse_response(self):
        with open(os.path.join(os.path.dirname(__file__), 'response.txt'), 'rb') as f:
            response = SemrushClient.parse_response(f.read())
            self.assertEqual(response.__class__, list)
            self.assertEqual(len(response), 10)

    @patch('requests.get')
    def test_domain_ranks(self, RequestsGet):
        contents = semrush_response_bytes('response.txt')

        RequestsGet.return_value = Response()
        RequestsGet.return_value.status_code = 200
        RequestsGet.return_value._content = contents

        s = SemrushClient(key='fdjsaiorghrtbnjvlouhsdlf')
        result = s.domain_ranks('example.com')
        self.assertEqual(len(result), 10)



