import json
from types import SimpleNamespace

_json_file = open('./config.json')
_json_content = _json_file.read()
CONFIG = json.loads(_json_content, object_hook=lambda d: SimpleNamespace(**d))
