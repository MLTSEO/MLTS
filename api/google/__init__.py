from __future__ import absolute_import, print_function, unicode_literals

import logging
import warnings

import httplib2
from apiclient import errors
from apiclient.discovery import build
from oauth2client.client import OAuth2WebServerFlow
from oauth2client.file import Storage
from urllib.error import HTTPError
logging.getLogger('googleapiclient.discovery_cache').setLevel(logging.ERROR)
logging.getLogger('oauth2client._helpers').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# Call GSC Service
def authenticate(config):

    # Check https://developers.google.com/webmaster-tools/search-console-api-original/v3/ for all available scopes
    OAUTH_SCOPE = config['OAUTH_SCOPE']

    # Redirect URI for installed apps
    REDIRECT_URI = 'urn:ietf:wg:oauth:2.0:oob'

    # Create a credential storage object.  You pick the filename.
    storage = Storage(config['GOOGLE_CREDENTIALS'])

    # Attempt to load existing credentials.  Null is returned if it fails.
    credentials = storage.get()

    # Only attempt to get new credentials if the load failed.
    if not credentials:

        # Run through the OAuth flow and retrieve credentials
        flow = OAuth2WebServerFlow(config['CLIENT_ID'], config['CLIENT_SECRET'], OAUTH_SCOPE, REDIRECT_URI)
        authorize_url = flow.step1_get_authorize_url()
        print ('Go to the following link in your browser: ' + authorize_url)
        code = input('Enter verification code: ').strip()
        credentials = flow.step2_exchange(code)
        storage.put(credentials)
        if storage.get():
            print('Credentials saved for later.')

    # Create an httplib2.Http object and authorize it with our credentials
    http = httplib2.Http()

    client = credentials.authorize(http)

    return client
