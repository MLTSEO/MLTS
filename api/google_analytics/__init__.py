# -*- coding: utf-8 -*-
"""Wrapper around the GA API."""

import httplib2
from apiclient.discovery import build
from googleanalytics import utils, account, auth
import addressable
from config import config
from .errors import *


class GaClient(object):

    def __init__(self, *kwargs):
        if len(kwargs) > 0:
            client = kwargs[0]
        if not isinstance(client,httplib2.Http):
            from api.google import authenticate
            try:
                client = authenticate()
            except:
                raise GaConfigError('Make sure that CLIENT_ID and CLIENT_SECRET is set in config.py')

        self.DATA_FOLDER = config['DATA_FOLDER']
        self.ROW_LIMIT = config['ROW_LIMIT']
        self.accounts = []
        self.client = client

        self.get_ga_service()

    # Call GA Service
    def get_ga_service(self):
        service = build('analytics', 'v3', http=self.client)
        raw_accounts = service.management().accounts().list().execute()['items']
        accounts = [account.Account(raw, service, self.client.credentials) for raw in raw_accounts]

        self.accounts = addressable.List(accounts, indices=['id', 'name'], insensitive=True)

        return addressable.List(accounts, indices=['id', 'name'], insensitive=True)


    def get_profile(self, account=None, webproperty=None, profile=None, default_profile=True):

        return auth.navigate(self.accounts, account=account, webproperty=webproperty, profile=profile, default_profile=default_profile)
