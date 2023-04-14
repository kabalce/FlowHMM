import json


def read_jsonl(f):
    txt = f.readlines()
    data = [json.loads(t) for t in txt]
    return data


def encode_event_type(string):
    if string == 'carts':
        return 0
    elif string == 'clicks':
        return 1
    elif string == 'orders':
        return 2
    else:
        return None
