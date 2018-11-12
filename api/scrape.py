import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

from goose3 import Goose
from bs4 import BeautifulSoup
from lxml import etree

import json
import pandas as pd
import urllib
from urllib.parse import urlparse
import re


def parse_html(html):

    article = None

    try:
        extractor = Goose()
        article = extractor.extract(raw_html=html)
        clean_text = article.cleaned_text
        print("Extracted {} words.".format( str(len(clean_text.split(' '))) ))

    except Exception as e:
        print ("Clean HTML Error:", e)
        clean_text = None

    return clean_text, article



def extract_url_data(url, **data):

    timeout = data.get('timeout', 10)
    render_endpoint = data.get('render_endpoint', '')
    user_agent = data.get('user_agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.77 Safari/537.36')

    render_url = render_endpoint + url

    try:
        print('loading url {}'.format(render_url))
        headers = {'user-agent': user_agent}
        response = requests.get(render_url, headers=headers, timeout=timeout)
        status = response.status_code
        html = response.text

        text, article = parse_html(html)

        infos = article.infos

        soup = BeautifulSoup(html, 'lxml')
        infos['h1'] = [x.text for x in soup.findAll("h1")]
        infos['h2'] = [x.text for x in soup.findAll("h2")]
        infos['html'] = html

    except Exception as e:
        print('Extract Text URL Error: ',str(e))
        return None, None

    return text, infos
