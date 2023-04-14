from clickstream_experiment.source.utils import encode_event_type
from abc import ABC
import json
from collections import defaultdict as dd


class ClickStreamEvent(ABC):
    def __init__(self, event, clickstream):
        self.item_id = event["aid"]
        self.ts = event["ts"]
        self.type = encode_event_type(event["type"])
        clickstream.item_ids[self.item_id] += 1


class ClickStreamSession(ABC):
    def __init__(self, session, clickstream):
        self.session_id = session["session"]
        self.event_list = [
            ClickStreamEvent(event, clickstream)
            for event in session["events"]
            if len(session["events"])
        ]
        clickstream.session_lengths[len(self.event_list)] += 1

    def list_items(self):
        return [event.item_id for event in self.event_list]


class ClickStream(ABC):
    def __init__(self, data):
        self.item_ids = dd(lambda: 0)
        self.session_lengths = dd(lambda: 0)
        self.sessions = [
            ClickStreamSession(json.loads(session), self)
            for _, session in enumerate(data)
        ]

    def item_sequences(self, min_len: int = 1):
        return [s.list_items() for s in self.sessions if len(s.event_list) >= min_len]
