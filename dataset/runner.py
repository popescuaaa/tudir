import torch
import numpy as np
import pandas as pd
import re
import pprint
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pp = pprint.PrettyPrinter(indent=2)

# Networking
import requests
from bs4 import BeautifulSoup

from threading import Thread
from typing import List


def orcas_to_raw_dataframe(path_file):
    """
    Reads the orcas dataset and returns a pandas dataframe.
    """
    f = open(path_file, 'r')
    lines = f.readlines()
    raw_df = pd.DataFrame(columns=['q_id', 'q', 'url_id', 'url'])
    for line in lines:
        re_gr = re.findall(r'(\d+)[\t ](.*)[\t ](D\d+)[\t ](.*)', line)[0]
        q_id, q, url_id, url = re_gr
        q.strip("\t ")
        url.strip("\t ")
        raw_df.loc[len(raw_df)] = [q_id, q, url_id, url]
    f.close()
    return raw_df


def orcas_to_agg_df(path):
    """
    Reads the orcas dataset and returns a pandas dataframe.
    """
    f = open(path, 'r')
    agg_df = pd.DataFrame(columns=['q_id', 'q', '[url_id, url]'])
    qid_urlid_dict = {}
    lines = f.readlines()
    for line in lines:
        re_gr = re.findall(r'(\d+)[\t ](.*)[\t ](D\d+)[\t ](.*)', line)[0]
        q_id, q, url_id, url = re_gr
        q.strip("\t ")
        url.strip("\t ")
        if (q_id, q) not in qid_urlid_dict:
            qid_urlid_dict[(q_id, q)] = [[url_id, url]]
        else:
            qid_urlid_dict[(q_id, q)].append([url_id, url])
    f.close()

    for key, value in qid_urlid_dict.items():
        agg_df.loc[len(agg_df)] = [key[0], key[1], value]
    # agg_df.set_index('q_id', inplace=True)
    return agg_df

if __name__ == "__main__":
    orcas_path = "./orcas_subset.tsv"
    orcas_df = orcas_to_agg_df(orcas_path)
    orcas_df.head(10)
    orcas_df["q_id"].nunique()

    for idx, row in orcas_df.iterrows():
        print(row['q'])
        content = ""
        for e in row['[url_id, url]']:
            url = e[1]
            r = requests.get(url)
            print(r.status_code)

            if r.status_code == 200:
                soup = BeautifulSoup(r.text, 'html.parser')
                for p in soup.find_all("p"):
                    content += p.text
                    content += "\n"
                
        # Write content to file
        f = open(f"./content/{row['q_id']}.txt", "w")
        f.write(content)
        f.close()