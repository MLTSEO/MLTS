
config= {
    # Google Cloud Credentials
    'CLIENT_ID': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX.apps.googleusercontent.com',
    'CLIENT_SECRET': 'XXXXXXXXXXXXXXXXXXXXXXXXXX',
    'OAUTH_SCOPE': ['https://www.googleapis.com/auth/webmasters.readonly','https://www.googleapis.com/auth/analytics.readonly'],
    'GOOGLE_CREDENTIALS': 'XXXXXXXXXXXXXXXXXXXXXX',

    # GSC data
    'ROW_LIMIT': 25000,
    'DATA_FOLDER': 'data',

    # SEMRush
    'SEMRUSH_KEY': 'XXXXXXXXXXXXXXXXXXXXXXXXXXXX',
    'SEMRUSH_API_URL': 'http://api.semrush.com/',

    # Proxy Service
    'PROXY_ENDPOINT': 'http://a-proxy-service/endpoint',

    'RANDOM_SEED': 12345,

    # IBM Watson
    'IBM_WATSON_CREDENTIALS':  {
                          "url": "https://gateway.watsonplatform.net/natural-language-understanding/api",
                          "version": "2018-03-19",
                          "username": "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",
                          "password": "XXXXXXXXXXXXXXXX"
                         }

}
