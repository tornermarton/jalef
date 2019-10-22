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
