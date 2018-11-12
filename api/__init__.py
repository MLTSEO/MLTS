# encoding: utf-8

from api.google_search_console import GscClient
from api.python_semrush import SEMRushClient
from api.google_analytics import GaClient
from api.watson import WatsonClient

from api.scrape import extract_url_data

from api.google import authenticate
from config import config

client = authenticate(config)
gaservice = GaClient(client)
semrushservice = SEMRushClient(config)
gscservice = GscClient(client)
watsonservice = WatsonClient(config)
