import requests
import time
from datetime import datetime


class Submission(object):
    def __init__(self, reddit_id, subreddit, symbol, name, title, content, timestamp):
        self.reddit_id = reddit_id
        self.subreddit = subreddit
        self.symbol = symbol
        self.name = name
        self.title = title
        self.content = content
        self.timestamp = timestamp

    def mysql_str(self):
        return ", ".join(["`{}`={}".format(str(key), str(value)) for key, value in self.__dict__.items()])

    def get_values_tuple(self):
        return tuple([str(x) for x in self.__dict__.values()])

    def __str__(self):
        return str(self.get_values_tuple())

    def __repr__(self):
        return self.__str__()


class PushshiftScraper(object):
    URL = "https://api.pushshift.io/reddit/search/"

    def __init__(self):
        self._results = None

    @staticmethod
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

    def get_submissions(self, query: str, fields: list = None, subreddits: list = None, max_submissions: int = 2000):

        url = PushshiftScraper.URL + "submission"

        if fields is None:
            fields = ["id", "subreddit", "title", "selftext", "created_utc"]
        submissions = []

        params = {"fields": ",".join(fields),
                  "size": "500",
                  "is_video": "false",
                  "selftext:not": "[removed]",
                  "q": query
                  }

        if subreddits is not None:
            params.update({"subreddit": ",".join(subreddits)})

        last_page = None

        while last_page != [] and len(submissions) < max_submissions:
            last_page = self.scrape_page(url, params, last_page)
            submissions += last_page
            time.sleep(2)

        results = [Submission(e["id"], e["subreddit"], query, e["title"], e["selftext"],
                                     datetime.fromtimestamp(e["created_utc"])) for e in submissions[:max_submissions]]

        return results


class Reddit(object):
    @staticmethod
    def PushshiftScraper():
        return PushshiftScraper()