from datetime import datetime
import requests, time
import numpy as np
import string
import mysql.connector
import pandas as pd
import re


class Submission(object):
    def __init__(self, reddit_id, subreddit, symbol, title, content, timestamp):
        self.reddit_id = reddit_id
        self.subreddit = subreddit
        self.symbol = symbol
        self.title = title
        self.content = content
        self.timestamp = timestamp

    def mysql_str(self):
        return ", ".join(["`{}`={}".format(key, value) for key, value in self.__dict__.items()])

    def get_values_tuple(self):
        return tuple(self.__dict__.values())


# End class Submission


class Cursor(object):
    def __init__(self, connection):
        self._connection = connection

        self._cursor = None

    def __enter__(self):
        if self._cursor is None:
            self._cursor = self._connection.cursor()
        else:
            raise ValueError("Cursor is initialized!")

        return self._cursor

    def __exit__(self, exc_type, exc_value, traceback):
        if self._cursor is not None:
            self._cursor.close()
            self._cursor = None
        else:
            raise ValueError("Cursor is not initialized!")

        return self


# End class Cursor


class DatabaseConnection(object):
    def __init__(self, host, user, password, database):
        self.__host = host
        self.__user = user
        self.__password = password
        self.__database = database

        self._connection = None

    def __enter__(self):
        self.connect()

        return self._connection

    def __exit__(self, exc_type, exc_value, traceback):
        self.disconnect()

        return self

    def connect(self):
        self._connection = mysql.connector.connect(
            host=self.__host,
            user=self.__user,
            passwd=self.__password,
            database=self.__database
        )

    def disconnect(self):
        self._connection.close()
        self._connection = None


# End class DatabaseConnection


def get_query_terms(n=10):
    slick = pd.read_html(requests.get("https://www.slickcharts.com/sp500").content)[0].drop(columns=['#'])
    wiki = pd.read_html(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies").content)[0]

    sp500 = slick.copy()

    banned = ["& Co\.", "inc\.", "plc", "corporation", "group", "Class .", "co\.", "corp", "corp\.", "Company", "inc",
              "n\.a\.", "ltd\."]
    pattern = re.compile("({})".format("|".join(banned)), re.I)

    sp500["Name"] = [pattern.sub("", x).strip() for x in sp500["Company"].values]

    sectors = [None] * len(sp500)
    sub_ind = [None] * len(sp500)

    print(len(sectors))

    for idx, row in sp500.iterrows():
        try:
            sectors[idx] = wiki[wiki["Symbol"] == sp500.iloc[idx]["Symbol"]]["GICS Sector"].values[0]
            sub_ind[idx] = wiki[wiki["Symbol"] == sp500.iloc[idx]["Symbol"]]["GICS Sub Industry"].values[0]
        except:
            continue

    sp500["GICS Sector"] = sectors
    sp500["GICS Sub Industry"] = sub_ind

    sp500 = sp500.sort_values("Weight", ascending=False).drop_duplicates(subset="Name")

    # remove single letter symbols (can't use the letter V as search term)
    short_symbols = []

    for idx, e in sp500.iterrows():
        s = e["Symbol"]

        if len(s) < 2 or e["GICS Sector"] is None:
            short_symbols.append(s)

    sp500 = sp500[~sp500["Symbol"].isin(short_symbols)].reset_index(drop=True)

    assert len(sp500[sp500["GICS Sector"] is None]) == 0

    return sp500[:n]["Symbol"].values, sp500[:n]["Name"].values


def scrape_page(url: str, params: dict, last_page: list = None):
    if last_page is not None:
        if len(last_page) > 0:
            # resume from where we left at the last page
            params["before"] = last_page[-1]["created_utc"]
        else:
            # the last page was empty, we are past the last page
            return []

    results = requests.get(url=url, params="&".join("%s=%s" % (k, v) for k, v in params.items()))

    if not results.ok:
        # something wrong happened
        raise Exception("Server returned status code {}".format(results.status_code))

    data = results.json()["data"]

    # remove empty results
    data = [e for e in data if e["selftext"] != ""]

    return data


def save_submissions(submissions):
    with DatabaseConnection(
            host="172.17.0.3",
            user="tm",
            password="FlDa3846",
            database="reddit"
    ) as connection:
        with Cursor(connection) as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS submissions (id INT AUTO_INCREMENT PRIMARY KEY, reddit_id VARCHAR(32) NOT NULL UNIQUE, subreddit VARCHAR(255), symbol VARCHAR(16), title TEXT, content TEXT, timestamp TIMESTAMP);")

            sql = "INSERT IGNORE INTO submissions (reddit_id, subreddit, symbol, title, content, timestamp) VALUES (%s, %s, %s, %s, %s, %s)"

            val = [e.get_values_tuple() for e in submissions]

            cursor.executemany(sql, val)

            connection.commit()

            print(cursor.rowcount, "was inserted.")


if __name__ == '__main__':
    url = "https://api.pushshift.io/reddit/search/submission"
    fields = [
        "id",
        "subreddit",
        "title",
        "selftext",
        "created_utc"
    ]
    subreddits = [
        "presonalfinance",
        "investing",
        "securityanalysis",
        "algotrading",
        "backupwallstreetbets",
        "Banking",
        "Daytrading",
        "Economics",
        "finance",
        "financialindependence",
        "Forex",
        "investing_discussion",
        "InvestmentClub",
        "options",
        "personalfinance",
        "portfolios",
        "StockMarket",
        "stocks",
        "ValueInvesting",
        "wallstreetbets"
    ]

    max_submissions_per_query = 5000

    symbols, names = get_query_terms(10)

    print("Symbols: {}".format(",".join(symbols)))

    for q, s in np.concatenate((symbols, names)), np.concatenate((symbols, symbols)):

        params = {"fields": ",".join(fields),
                  "size": "500",
                  "is_video": "false",
                  "selftext:not": "[removed]",
                  "subreddit": ",".join(subreddits),
                  "q": q
                  }

        last_page = None

        n_submissions = 0

        while last_page != [] and n_submissions < max_submissions_per_query:
            try:
                last_page = scrape_page(url, params, last_page)

                n_submissions += len(last_page)

                submissions = [Submission(e["id"], e["subreddit"], s, e["title"], e["selftext"],
                                          datetime.fromtimestamp(e["created_utc"])) for e in last_page]

                save_submissions(submissions)

                time.sleep(2)
            except Exception as e:
                print(e)
