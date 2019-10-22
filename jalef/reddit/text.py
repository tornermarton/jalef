import re
import string
from bs4 import BeautifulSoup
from markdown import markdown


class Text(object):
    def __init__(self, text, original=None):
        self.text = str(text)

        if original is None:
            original = self.text
        self.__original = str(original)

    def remove_linebreaks(self):
        return Text(self.text.replace("\n", " ").replace("\r", " "), self.__original)

    def remove_multiple_spaces(self):
        return Text(re.sub(' +', ' ', self.text), self.__original)

    def remove_numbers(self):
        return Text("".join(c for c in self.text if not c.isdigit()), self.__original)

    def remove_special_characters(self, dots_and_commas_also: bool = False):
        if dots_and_commas_also:
            return Text(''.join(c for c in self.text if c not in string.punctuation), self.__original)
        else:
            return Text(re.sub('[^.,a-zA-Z \.]', '', self.text), self.__original)

    def trim(self):
        return Text(self.text.strip(), self.__original)

    def remove_links(self):
        return Text(
            re.sub(
                r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
                " ",
                self.text
            ),
            self.__original
        )

    def remove_markdown(self):
        # md -> html -> text since BeautifulSoup can extract text cleanly
        html = markdown(self.text)

        # remove code snippets
        html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
        html = re.sub(r'<code>(.*?)</code >', ' ', html)

        # extract text
        soup = BeautifulSoup(html, "html.parser")
        text = ''.join(soup.findAll(text=True))

        return Text(text, self.__original)

    def lowercase(self):
        return Text(self.text.lower(), self.__original)

    def clean(self):
        return self.remove_linebreaks()\
            .remove_markdown()\
            .remove_special_characters()\
            .remove_numbers()\
            .remove_multiple_spaces()\
            .trim()\
            .lowercase()

    def remove_word(self, word):
        return Text(self.text.replace(' '+word+' ', ' '), self.__original)

    def original(self):
        return self.__original

    def __str__(self):
        return self.text

    def __repr__(self):
        return str(self)
