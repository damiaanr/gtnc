from src import *
import itertools

e = Evaluator(load_from_cache=True, ev_params={'normalise': False})

for l1, l2 in itertools.combinations(e.languages, r=2):
    score = e.scores[(l1, l2)]
    if score > 0:
        print('%s <-> %s: %.3f' % (l1, l2, score))
