from __future__ import absolute_import, print_function, unicode_literals

import sys
import httplib2
import pandas as pd
import time
import re
from tqdm import tqdm
import datetime as dt
import os
from datetime import timedelta, date
from urllib.error import HTTPError
from apiclient import errors
from apiclient.discovery import build

from config import config
from .errors import *

class GscClient(object):

    def __init__(self, *kwargs):
        if len(kwargs) > 0:
            client = kwargs[0]
        if not isinstance(client,httplib2.Http):
            from api.google import authenticate
            try:
                client = authenticate()
            except:
                raise GscConfigError('Make sure that CLIENT_ID and CLIENT_SECRET is set in config.py')

        self.DATA_FOLDER = config['DATA_FOLDER']
        self.ROW_LIMIT = config['ROW_LIMIT']
        self.client = client

    # Call GSC Service
    def get_gsc_service(self):

        webmasters_service = build('webmasters', 'v3', http=self.client)

        return webmasters_service

    @staticmethod
    def daterange(start_date, end_date):
        for n in range(int ((end_date - start_date).days)):
            yield start_date + timedelta(n)

    @staticmethod
    def execute_request(service, property_uri, request):
        """
          Executes a searchAnalytics.query request.
          Args:
            service: The webmasters service to use when executing the query.
            property_uri: The site or app URI to request data for.
            request: The request to be executed.
          Returns:
            An array of response rows.
        """
        return service.searchanalytics().query(
          siteUrl=property_uri, body=request).execute()
    
    
    
    '''
    Parameters:

    Positional:
    clienturl: (str) The domain URL property name in Google Search Console.
    days_back: (int) How many days history to pull.

    Keyword:
    thresholdtype: (str)  'click' or 'impression'. Default: impression
    threshold: (int) Keep pulling, daily until less than this number of either clicks or impressions. Default: 1
    poslimit: (int) Omit results above this limit. Default: None
    country: (str) Country. Default: usa
    outputFn: (str) Name of the output file.  If not set, a unique name will be chosen.
    '''
    def get_site_data(self, clienturl, days, **data):

        thresholdtype = data.get('threshold_type', 'impression')
        threshold = data.get('threshold', 1)
        poslimit = data.get('pos_limit', None)
        country = data.get('country', 'usa')
        outputFn = data.get('output_fn', "".join([self.DATA_FOLDER, "/", "gsc_", re.sub('[^0-9a-zA-Z]+', '_', clienturl), dt.date.today().strftime("%Y_%m"), ".csv"]))

        if (self.DATA_FOLDER + "/") not in outputFn and os.path.isdir(self.DATA_FOLDER):
            outputFn = "".join([self.DATA_FOLDER, "/",outputFn])

        start_date = (dt.date.today()-dt.timedelta(days = (days+3) ))
        end_date = (dt.date.today()-dt.timedelta(days = 3))

        row_limit = self.ROW_LIMIT

        if os.path.isfile(outputFn):
            print('Reloading Existing: ' + outputFn)
            df = pd.read_csv(outputFn, encoding = "utf-8")
            if poslimit is not None:
                return df[df.position <= poslimit]
            return df

        output = []

        print("Building new {} file".format(outputFn));
        print('Getting Webmaster Service')
        webmasters_service = self.get_gsc_service()
        time.sleep(1)

        pbar = tqdm(total=int((end_date - start_date).days), desc='Pulling Google Search Console Data', file=sys.stdout)

        for single_date in self.daterange(start_date, end_date):

            month_date = str(single_date.strftime("%Y-%m"))
            single_date = str(single_date)
            pbar.update()

            try:
                    n = 0
                    Count = 11
                    startRow = 0
                    while (Count >= threshold):

                        #print("-----Executing------- " + str(startRow))
                        request = {
                            'startDate': single_date,
                            'endDate': single_date,
                            'dimensions': ['query', 'page'],
                             'dimensionFilterGroups': [
                              {
                               'filters': [
                                {
                                 'dimension': 'country',
                                 'expression': country
                                }
                               ]
                              }
                              ],
                            'rowLimit': row_limit,
                            'startRow': int(startRow)
                        }
                        try:
                            response = self.execute_request(webmasters_service, clienturl, request)
                        except Exception as e:
                            print("API Error:", str(e))
                            time.sleep(30)
                            continue

                        startRow = startRow + (row_limit)
                        tCount, NewOutput = self.handle_response(response, clienturl, thresholdtype, threshold, month_date)
                        output = output + NewOutput

                        n = n + 1
                        if (n % 3 == 0):
                            time.sleep(1)
                        Count = int(tCount)


            except Exception as e:
                    raise GscApiError(str(e))


        pbar.close()

        df = pd.DataFrame(output)
        print("Total rows found: {}. Saving to csv.".format(str( len(df) ) ) );
        df.to_csv(outputFn, header=True, index=False, encoding='utf-8')

        if poslimit:
            return df[df.position <= poslimit]

        return df

    @staticmethod
    def handle_response(response, clienturl, thresholdtype, threshold, month_date):

        output = []
        tCount = -1

        if 'rows' not in response:
            return int(tCount), output

        rows = response['rows']
        row_format = '{:<20}' + '{:>20}' * 4
        for row in rows:
            keys = ''

            if 'keys' in row:

                if thresholdtype == 'click':
                    tcheck = int(row['clicks'])
                else:
                    tcheck = int(row['impressions'])

                if tcheck < int(threshold):
                    continue

                query = str(row['keys'][0])
                page = str(row['keys'][1])
                dict = {'clientID': clienturl, 'query': query, 'page': page,
                        'clicks': row['clicks'], 'impressions': row['impressions'], 'ctr': row['ctr'],
                        'position': int(row['position']), 'month':str(month_date)}

                output.append(dict)
                tCount = tcheck

        return int(tCount), output
