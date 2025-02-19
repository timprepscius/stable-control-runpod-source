import json
import traceback

from rp_handler import process

with open("test_input.json", "r") as f:
    test_input_json = json.load(f)

test_input_json["id"] = "test_id"

try:
    result = process(test_input_json['id'], test_input_json['input'])
except Exception as e:
    traceback.print_exc()