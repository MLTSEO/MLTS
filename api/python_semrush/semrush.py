# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function, unicode_literals
import requests
import pandas as pd
from .errors import *

SEMRUSH_ASNS_API_URL = 'http://api.asns.backend.semrush.com/'
SEMRUSH_API_V1_URL = 'http://api.semrush.com/analytics/v1/'

REGIONAL_DATABASES = {
    'google.com': 'us',
    'google.co.uk': 'uk',
    'google.ca': 'ca',
    'google.ru': 'ru',
    'google.de': 'de',
    'google.fr': 'fr',
    'google.es': 'es',
    'google.it': 'it',
    'google.com.br': 'br',
    'google.com.au': 'au',
    'bing.com': 'bing-us',
    'google.com.ar': 'ar',
    'google.be': 'be',
    'google.ch': 'ch',
    'google.dk': 'dk',
    'google.fi': 'fi',
    'google.com.hk': 'hk',
    'google.ie': 'ie',
    'google.co.il': 'il',
    'google.com.mx': 'mx',
    'google.nl': 'nl',
    'google.no': 'no',
    'google.pl': 'pl',
    'google.se': 'se',
    'google.com.sg': 'sg',
    'google.com.tr': 'tr',
    'm.google.com': 'mobile-us',
    'google.co.jp': 'jp',
    'google.co.in': 'in'
}


class SEMRushClient(object):

    def __init__(self, config):
        if not config:
            raise SemRushKeyError('No config data provided.')

        self.api_url = config['SEMRUSH_API_URL']
        self.key = config['SEMRUSH_KEY']

    @staticmethod
    def get_database_from_search_engine(search_engine='google.com'):
        if search_engine in REGIONAL_DATABASES:
            return REGIONAL_DATABASES[search_engine]
        else:
            raise SemRushRegionalDatabaseError('%s - is not an accepted search engine.' % search_engine)

    # Report producing methods
    def produce(self, report_type, **kwargs):
        data = self.retrieve(report_type, **kwargs)
        return self.parse_response(data)

    def retrieve(self, report_type, **kwargs):
        kwargs['type'] = report_type
        kwargs['key'] = self.key

        response = requests.get(self.api_url, params=kwargs)

        if response.status_code == 200:
            return response.content
        else:
            raise BaseSemrushError(response.content)

    @staticmethod
    def parse_response(data):
        results = []
        data = data.decode('unicode_escape')
        lines = data.split('\r\n')
        lines = list(filter(bool, lines))
        columns = lines[0].split(';')

        for line in lines[1:]:
            result = {}
            for i, datum in enumerate(line.split(';')):
                result[columns[i]] = datum.strip('"\n\r\t')
            results.append(result)

        return pd.DataFrame(results)

    # Overview Reports
    def domain_ranks(self, domain, **kwargs):
        """
        Domain Overview (All Databases)
        This report provides live or historical data on a domain’s keyword rankings in both organic and paid search in
        all regional databases.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_date: date in format "YYYYMM15"
        - export_columns: Db, Dn, Rk, Or, Ot, Oc, Ad, At, Ac
        """
        return self.produce('domain_ranks', domain=domain, **kwargs)

    def domain_rank(self, domain, database, **kwargs):
        """
        Domain Overview (One Database)
        This report provides live or historical data on a domain’s keyword rankings in both organic and paid search in a
        chosen regional database.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - export_escape: 1 to wrap report columns in quotation marks (")
        - export decode: 1 or 0, 0 to url encode string
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Rk, Or, Ot, Oc, Ad, At, Ac
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_rank', domain=domain, database=database, **kwargs)

    def domain_rank_history(self, domain, database, **kwargs):
        """
        Domain Overview (History)
        This report provides live and historical data on a domain’s keyword rankings in both organic and paid search in
        a chosen database.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_daily: 1
        - export_columns: Rk, Or, Ot, Oc, Ad, At, Ac, Dt
        - display_sort: dt_asc, dt_desc
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_rank_history', domain=domain, database=database, **kwargs)

    def rank_difference(self, database, **kwargs):
        """
        Winners and Losers
        This report shows changes in the number of keywords, traffic, and budget estimates of the most popular websites
        in Google's top 20 and paid search results.

        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Rk, Or, Ot, Oc, Ad, At, Ac, Om, Tm, Um, Am, Bm, Cm
        - display_sort: om_asc, om_desc, tm_asc, tm_desc, um_asc, um_desc, am_asc, am_desc, bm_asc, bm_desc, cm_asc,
            cm_desc
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('rank_difference', database=database, **kwargs)

    def rank(self, database, **kwargs):
        """
        Semrush Rank
        This report lists the most popular domains ranked by traffic originating from Google's top 20 organic search
        results.

        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Rk, Or, Ot, Oc, Ad, At, Ac
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('rank', database=database, **kwargs)

    # Domain Reports
    def domain_organic(self, domain, database, **kwargs):
        """
        Domain Organic Search Keywords
        This report lists keywords that bring users to a domain via Google's top 20 organic search results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Po, Pp, Pd, Nq, Cp, Ur, Tr, Tc, Co, Nr, Td
        - display_sort: tr_asc, tr_desc, po_asc, po_desc, tc_asc, tc_desc
        - display_positions: new, lost, rise or fall
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_organic', domain=domain, database=database, **kwargs)

    def domain_adwords(self, domain, database, **kwargs):
        """
        Domain Paid Search
        This report lists keywords that bring users to a domain via Google's paid search results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Po, Pp, Pd, Ab, Nq, Cp, Tr, Tc, Co, Nr, Td, Tt, Ds, Vu, Ur
        - display_sort: tr_asc, tr_desc, po_asc, po_desc, tc_asc, tc_desc
        - display_positions: new, lost, rise or fall
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_adwords', domain=domain, database=database, **kwargs)

    def domain_adwords_unique(self, domain, database, **kwargs):
        """
        Ads Copies
        This report shows unique ad copies SEMrush noticed when the domain ranked in Google's paid search results for
        keywords from our databases.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Ph, Po, Pp, Nq, Cp, Tr, Tc, Co, Nr, Td, Tt, Ds, Vu, Ur, Pc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_adwords_unique', domain=domain, database=database, **kwargs)

    def domain_organic_organic(self, domain, database, **kwargs):
        """
        Competitors In Organic Search
        This report lists a domain’s competitors in organic search results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Cr, Np, Or, Ot, Oc, Ad
        - display_sort: np_desc, cr_desc
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_organic_organic', domain=domain, database=database, **kwargs)

    def domain_adwords_adwords(self, domain, database, **kwargs):
        """
        Competitors In Paid Search
        This report lists a domain’s competitors in paid search results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Cr, Np, Ad, At, Ac, Or
        - display_sort: np_desc, cr_desc
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_adwords_adwords', domain=domain, database=database, **kwargs)

    def domain_adwords_historical(self, domain, database, **kwargs):
        """
        Domains Ads History
        This report shows keywords a domain has bid on in the last 12 months and its positions in paid search results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Ph, Dt, Po, Cp, Nq, Tr, Ur, Tt, Ds, Vu, Cv
        - display_sort: cv_asc, cv_desc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_adwords_historical', domain=domain, database=database, **kwargs)

    def domain_domains(self, domains, database, **kwargs):
        """
        Domain Vs. Domain
        This report allows users to compare up to five domains by common keywords, unique keywords, all keywords, or
        search terms that are unique to the first domain.

        :param domains: The domains to query data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, P0, P1, P2, P3, P4, Nr, Cp, Co, Nq
        - display_sort: p0_asc, p0_desc, p1_asc, p1_desc, p2_asc, p2_desc, p3_asc, p3_desc, p4_asc, p4_desc, nq_asc,
            nq_desc, co_asc, co_desc, cp_asc, cp_desc, nr_asc, nr_desc
        - display_filter:

        Note: Refer to SEMrush API documentation for format of domains
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_domains', domains=domains, database=database, **kwargs)

    def domain_shopping(self, domain, database, **kwargs):
        """
        Domain PLA Search Keywords
        This report lists keywords that trigger a domain’s product listing ads to appear in Google's paid search
        results.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Ph, Po, Pp, Pd, Ab, Nq, Cp, Tr, Tc, Co, Nr, Td, Tt, Ds, Vu, Ur
        - display_sort: tr_asc, tr_desc, po_asc, po_desc, tc_asc, tc_desc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_shopping', domain=domain, database=database, **kwargs)

    def domain_shopping_unique(self, domain, database, **kwargs):
        """
        PLA Copies
        This report shows product listing ad copies SEMrush noticed when the domain ranked in Google's paid search results
        for keywords from our databases.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Ph, Po, Pp, Nq, Cp, Tr, Tc, Co, Nr, Td, Tt, Ds, Vu, Ur, Pc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_shopping_unique', domain=domain, database=database, **kwargs)

    def domain_shopping_shopping(self, domain, database, **kwargs):
        """
        PLA Competitors
        This report lists domains a queried domain is competing against in Google's paid search results with product
        listing ads.

        :param domain: The domain to query data for ie. 'example.com'
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Dn, Cr, Np, Ad, At, Ac, Or
        - display_sort: np_asc, np_desc, cr_asc, cr_desc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('domain_shopping_shopping', domain=domain, database=database, **kwargs)

    # Keyword Reports
    def phrase_all(self, phrase, **kwargs):
        """
        Keyword Overview (All Databases)
        This report provides a summary of a keyword, including its volume, CPC, competition, and the number of results
        in all regional databases.

        :param phrase: The phrase or term to obtain data for

        Optional kwargs
        - database:
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Db, Ph, Nq, Cp, Co
        """
        return self.produce('phrase_all', phrase=phrase, **kwargs)

    def phrase_this(self, phrase, database, **kwargs):
        """
        Keyword Overview (One Database)
        This report provides a summary of a keyword, including its volume, CPC, competition, and the number of results
        in a chosen regional database.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Nq, Cp, Co, Nr
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_this', phrase=phrase, database=database, **kwargs)

    def phrase_organic(self, phrase, database, **kwargs):
        """
        Organic Results
        This report lists domains that are ranking in Google's top 20 organic search results with a requested keyword.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Ur
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_organic', phrase=phrase, database=database, **kwargs)

    def phrase_adwords(self, phrase, database, **kwargs):
        """
        Paid Results
        This report lists domains that are ranking in Google's paid search results with a requested keyword.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Dn, Ur
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_adwords', phrase=phrase, database=database, **kwargs)

    def phrase_related(self, phrase, database, **kwargs):
        """
        Related Keywords
        This report provides an extended list of related keywords, synonyms, and variations relevant to a queried term
        in a chosen database.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Nq, Cp, Co, Nr, Td
        - display_sort: nq_asc, nq_desc, cp_asc, cp_desc, co_asc, co_desc, nr_asc, nr_desc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_related', phrase=phrase, database=database, **kwargs)

    def phrase_adwords_historical(self, phrase, database, **kwargs):
        """
        Keywords Ads History
        This report shows domains that have bid on a requested keyword in the last 12 months and their positions in paid
        search results.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Dn, Dt, Po, Ur, Tt, Ds, Vu, At, Ac, Ad
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_adwords_historical', database=database, **kwargs)

    def phrase_fullsearch(self, phrase, database, **kwargs):
        """
        Phrase Match Keywords
        The report offers a list of phrase matches and alternate search queries, including particular keywords or
        keyword expressions.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: Ph, Nq, Cp, Co, Nr, Td
        - display_sort: nq_asc, nq_desc, cp_asc, cp_desc, co_asc, co_desc, nr_asc, nr_desc
        - display_filter:
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_fullsearch', phrase=phrase, database=database, **kwargs)

    def phrase_kdi(self, phrase, database, **kwargs):
        """
        Keyword Difficulty
        This report provides keyword difficulty, an index that helps to estimate how difficult it would be to seize
        competitors' positions in organic search within the Google's top 20 with an indicated search term.

        :param phrase: The phrase or term to obtain data for
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - export_escape: 1
        - export_columns: Ph, Kd
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('phrase_kdi', phrase=phrase, database=database, **kwargs)

    # URL Reports
    def url_organic(self, url, database, **kwargs):
        """
        URL Organic Search Keywords
        This report lists keywords that bring users to a URL via Google's top 20 organic search results.

        :param url: The URL to obtain data for, ie. http://example.com
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Po, Nq, Cp, Co, Tr, Tc, Nr, Td
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('url_organic', url=url, database=database, **kwargs)

    def url_adwords(self, url, database, **kwargs):
        """
        URL Paid Search Keywords
        This report lists keywords that bring users to a URL via Google's paid search results.

        :param url: The URL to obtain data for, ie. http://example.com
        :param database: The database to query, one of the choices from REGIONAL_DATABASES

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - display_date: date in format "YYYYMM15"
        - export_columns: Ph, Po, Nq, Cp, Co, Tr, Tc, Nr, Td, Tt, Ds
        """
        if database not in REGIONAL_DATABASES.values():
            raise SemRushRegionalDatabaseError('%s - is not an accepted database.' % database)
        return self.produce('url_adwords', url=url, database=database, **kwargs)

    # Display Advertising Reports
    def publisher_text_ads(self, domain, **kwargs):
        """
        Publisher Display Ads
        This report lists display ads that have appeared on a publisher’s website.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: title, text, first_seen, last_seen, times_seen, avg_position, media_type, visible_url
        - display_sort: last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_asc,
            times_seen_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('publisher_text_ads', domain=domain, **kwargs)

    def publisher_advertisers(self, domain, **kwargs):
        """
        Advertisers
        This report lists advertisers whose display ads have appeared on a queried publisher’s website.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: domain, ads_count, first_seen, last_seen, times_seen, ads_percent
        - display_sort: last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_desc
            ads_count_asc, ads_count_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('publisher_advertisers', domain=domain, **kwargs)

    def advertiser_publishers(self, domain, **kwargs):
        """
        Publishers
        This report lists publisher’s websites where an advertiser’s display ads have appeared.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: domain, ads_count, first_seen, last_seen, times_seen, ads_percent
        - display_sort: last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_desc
            ads_count_asc, ads_count_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('advertiser_publishers', domain=domain, **kwargs)

    def advertiser_text_ads(self, domain, **kwargs):
        """
        Advertiser Display Ads
        This report lists display ads of a queried advertiser’s website.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: title,text, first_seen, last_seen, times_seen, avg_position, media_type, visible_url
        - display_sort: last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_asc,
            times_seen_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('advertiser_text_ads', domain=domain, **kwargs)

    def advertiser_landings(self, domain, **kwargs):
        """
        Landing Pages
        This report lists URLs of a domain’s landing pages promoted via display ads.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: target_url, first_seen, last_seen, times_seen, ads_count
        - display_sort: ast_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_desc,
            ads_count_asc, ads_count_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('advertiser_landings', domain=domain, **kwargs)

    def advertiser_publisher_text_ads(self, domain, **kwargs):
        """
        Advertiser Display Ads On A Publishers Website
        This report lists the display ads of a given advertiser that have appeared on a particular publisher’s website.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_escape: 1
        - export_decode: 1 or 0
        - export_columns: title,text, first_seen, last_seen, times_seen, avg_position, media_type, visible_url
        - display_sort	last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc, times_seen_asc, times_seen_asc,
            times_seen_desc
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        - display_filter:
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('advertiser_publisher_text_ads', domain=domain, **kwargs)

    def advertiser_rank(self, domain, **kwargs):
        """
        Advertisers Rank
        This report lists advertisers ranked by the total number of display ads noticed by SEMrush.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - export_escape: 1
        - export_columns: domain, ads_overall, text_ads_overall, ads_count, text_ads_count, times_seen, first_seen,
            last_seen, media_ads_overall, media_ads_count, publishers_overall, publishers_count
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('advertiser_rank', domain=domain, **kwargs)

    def publisher_rank(self, domain, **kwargs):
        """
        Publishers Rank
        This report lists publishers ranked by the total number of display ads noticed by SEMrush.

        :param domain: The domain to query data for ie. 'example.com'

        Optional kwargs
        - export_escape: 1
        - export_columns: domain, ads_overall, text_ads_overall, ads_count, text_ads_count, times_seen, first_seen,
            last_seen, media_ads_overall, media_ads_count, advertiser_overall, advertiser_count
        - device_type: all, desktop, smartphone_apple, smartphone_android, tablet_apple, tablet_android
        """
        kwargs['action'] = 'report'
        kwargs['export'] = 'api'
        self.api_url = SEMRUSH_ASNS_API_URL
        return self.produce('publisher_rank', domain=domain, **kwargs)

    # Backlinks
    def backlinks_overview(self, target, target_type='root_domain'):
        """
        Backlinks Overview
        This report provides a summary of backlinks, including their type, referring domains and IP addresses for a
        domain, root domain, or URL.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Kwargs
        - target_type: domain, root_domain or url
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_overview', target=target, target_type=target_type)

    def backlinks(self, target, target_type='root_domain', **kwargs):
        """
        Backlinks
        This report lists backlinks for a domain, root domain, or URL.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_columns: page_score, response_code, source_size, external_num, internal_num, redirect_url, source_url,
            source_title, image_url, target_url, target_title, anchor, image_alt, last_seen, first_seen, nofollow, form,
            frame, image, sitewide
        - display_sort: last_seen_asc, last_seen_desc, first_seen_asc, first_seen_desc
        - display_filter:
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks', target=target, target_type=target_type, **kwargs)

    def backlinks_refdomains(self, target, target_type='root_domain', **kwargs):
        """
        Referring Domains
        This report lists domains pointing to the queried domain, root domain, or URL.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_columns: domain_score, domain, backlinks_num, ip, country, first_seen, last_seen
        - display_sort: rank_asc, rank_desc, backlinks_asc, backlinks_desc, last_seen_asc, last_seen_desc, first_seen_asc,
            first_seen_desc
        - display_filter:
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_refdomains', target=target, target_type=target_type, **kwargs)

    def backlinks_refips(self, target, target_type='root_domain', **kwargs):
        """
        Referring IPs
        This report lists IP addresses where backlinks to a domain, root domain, or URL are coming from.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - display_limit: integer
        - display_offset: integer
        - export_columns: ip, country, domains_num, backlinks_num, first_seen, last_seen
        - display_sort: backlinks_num_asc, backlinks_num_desc, last_seen_asc, last_seen_desc, first_seen_asc,
            first_seen_desc, domains_num_asc domains_num_desc
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_refips', target=target, target_type=target_type, **kwargs)

    def backlinks_tld(self, target, target_type='root_domain', **kwargs):
        """
        TLD Distribution
        This report shows referring domain distributions depending on their top-level domain type.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - export_columns: zone, domains_num, backlinks_num
        - display_sort: backlinks_num_asc, backlinks_num_desc, domains_num_asc domains_num_desc
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_tld', target=target, target_type=target_type, **kwargs)

    def backlinks_geo(self, target, target_type='root_domain', **kwargs):
        """
        Referring Domains By Country
        This report shows referring domain distributions by country (an IP address defines a country).

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - export_columns: country, domains_num, backlinks_num
        - display_sort: backlinks_num_asc, backlinks_num_desc, domains_num_asc domains_num_desc
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_geo', target=target, target_type=target_type, **kwargs)

    def backlinks_anchors(self, target, target_type='root_domain', **kwargs):
        """
        Anchors
        This report lists anchor texts used in backlinks leading to the queried domain, root domain, or URL. It also
        includes the number of backlinks and referring domains per anchor.

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - export_columns: anchor, domains_num, backlinks_num, first_seen, last_seen
        - display_sort: backlinks_num_asc, backlinks_num_desc, last_seen_asc, last_seen_desc, first_seen_asc,
            first_seen_desc, domains_num_asc domains_num_desc
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_anchors', target=target, target_type=target_type, **kwargs)

    def backlinks_pages(self, target, target_type='root_domain', **kwargs):
        """
        Indexed Pages
        This report shows indexed pages of the queried domain

        :param target: A domain, root domain, or URL address to retrieve the data for ie.
        :param target_type: domain, root_domain or url

        Optional kwargs
        - export_columns: response_code, backlinks_num, domains_num, last_seen, external_num, internal_num, source_url,
            source_title
        - display_sort: backlinks_num_asc, backlinks_num_desc, domains_num_asc, domains_num_desc, last_seen_asc,
            last_seen_desc
        """
        self.api_url = SEMRUSH_API_V1_URL
        return self.produce('backlinks_pages', target=target, target_type=target_type, **kwargs)
