import json
from types import SimpleNamespace

with open('./config.json') as _json_file:
    _json_dict = json.load(_json_file)
_json_dict = '{"name": "John Smith", "hometown": {"name": "New York", "id": 123}}'
CONFIG = json.loads(_json_dict, object_hook=lambda d: SimpleNamespace(**d))
