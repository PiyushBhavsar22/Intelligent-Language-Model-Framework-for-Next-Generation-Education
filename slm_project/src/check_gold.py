import sys
sys.path.insert(0, ".")
from config import CONFIG
import evaluate as ev

g = ev.load_gold(CONFIG)
print(f"{len(g)} gold items, {sum(x['unanswerable'] for x in g)} unanswerable")