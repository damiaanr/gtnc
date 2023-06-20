from collections.abc import MutableMapping


class Scores(MutableMapping):
    """
    Scores object which functions like a dict with a tuple consiting of
    two languages as its keys, but allows for exchangeable language
    codes (ISO639-3). See: https://stackoverflow.com/a/41259637.
    """
    def __init__(self, arg=None):
        self.scores = {}

    def __getitem__(self, key):
        return self.scores[frozenset(key)]

    def __setitem__(self, key, value):
        self.scores[frozenset(key)] = value

    def __delitem__(self, key):
        del self.scores[frozenset(key)]

    def __iter__(self):
        return iter(self.scores)

    def __len__(self):
        return len(self.scores)
