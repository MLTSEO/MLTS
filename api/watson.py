from watson_developer_cloud import NaturalLanguageUnderstandingV1
from watson_developer_cloud.natural_language_understanding_v1 import Features, EntitiesOptions, KeywordsOptions
import pandas as pd
from config import config


class BaseWatsonError(Exception):
	pass

class WatsonConfigError(BaseWatsonError):
	pass

class WatsonApiError(BaseWatsonError):
	pass


class WatsonClient(object):

	def __init__(self, config):

		if not config or 'IBM_WATSON_CREDENTIALS' not in config:
			raise WatsonConfigError('No config data provided. Make sure that IBM_WATSON_CREDENTIALS is set in config.py')

		credentials = config['IBM_WATSON_CREDENTIALS']

		try:
			self.natural_language_understanding = NaturalLanguageUnderstandingV1 (
			version=credentials['version'],
			username= credentials['username'],
			password=credentials['password']
			)

		except Exception as e:
			raise WatsonApiError(str(e))

	def watson_keywords(self, html, **data):

		try:
			response = self.natural_language_understanding.analyze(html=html,features=Features(keywords=KeywordsOptions()))
			if "keywords" in response:
				keywords = response["keywords"]
				return pd.DataFrame(keywords)
			else:
				return pd.DataFrame()

		except Exception as e:
			raise WatsonApiError(str(e))

	def watson_entities(self, html, **data):

		try:
			response = self.natural_language_understanding.analyze(html=html,features=Features(entities=EntitiesOptions()))
			if "entities" in response:
				entities = response["entities"]
				return pd.DataFrame(entities)
			else:
				return pd.DataFrame()

		except Exception as e:
			raise WatsonApiError(str(e))
