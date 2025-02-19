import json

from rp_handler import process

with open("test_input.json", "r") as f:
    test_input_json = json.load(f)

test_input_json["id"] = "test_id"

result = process(test_input_json)
