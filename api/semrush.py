# -*- coding: utf-8 -*-
from python_semrush.semrush import SemrushClient
from urllib.parse import urlparse
from textdistance import distance
import pandas as pd

import config as cfg
import lib.scrape as scrape

client = SemrushClient(key=cfg.SEMRUSH_KEY)


    


def parseTerm(query, **data):
    
    numpos = data.get('pos_limit', 30)
    num = data.get('num', 10)
    exclusions = data.get('exclusions', cfg.GOOGLE_EXCLUSIONS)

    data = {}
    urls = []
    
    status, gdata = scrape.load_google_search(query, **data)
    
    if not status == 200:
        print('Incorrect status from Google Results (parseTerm): {}'.format(str(status)))
        return pd.DataFrame(), gdata

    # Get URLs
    response = [x['href'] for x in gdata['links']]


    if len(exclusions) > 0:
        for i,url in enumerate(response):
            skip = False
            for e in exclusions:
                if e in url:
                    skip = True
            if not skip:
                urls.append(url)
                num -= 1
            else:
                #print('Skipped URL:',url)
                pass
            if num < 1:
                break
    else:
        urls = response[:num]


    for url in urls:
        ranking = client.url_organic(url=url, database='us',export_columns='Ph, Po, Nq, Cp, Co, Tr, Tc, Nr, Td')
        if isinstance(ranking, list):
            #print('found',len(ranking), 'results for', url)
            for r in ranking:

                domain = " ".join(urlparse(url).hostname.split('.')[-2:])
                nontld = " ".join(urlparse(url).hostname.split('.')[-2:-1])
                ed_thresh = cfg.BRAND_ED_THRESHOLD
                
                brand = distance.s(domain,r['Keyword']) < ed_thresh or distance.s(nontld,r['Keyword']) < ed_thresh or nontld in r['Keyword'].split()

                #print(domain,r['Keyword'],distance.s(domain,r['Keyword']), brand)

                if int(r['Position']) > numpos:
                    continue

                if r['Keyword'] in data:
                    data[r['Keyword']]['count'] += 1
                    if brand:
                        data[r['Keyword']]['brand'] = 'Yes'
                else:
                    data[r['Keyword']] = {'volume'      : int(r['Search Volume']),
                                          'cpc'         : float(r['CPC']),
                                          'count'       : 1,
                                          'competition' : float(r['Competition']),
                                          'brand'       : "No"
                                         }
                    if brand:
                        data[r['Keyword']]['brand'] = 'Yes'
                        
                        
    values = []   
    for kw in data:
        val = data[kw]
        val['query'] = kw
        values.append(val)

    df = pd.DataFrame(values)

    return df, gdata
