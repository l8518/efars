import requests
import datetime
import json

from app.evaluation.run.abstract_receiver_adapter import AbstractReceiverAdapter


class BasicRecommenderReceiverAdapter(AbstractReceiverAdapter):
    def __init__(self):
        self.URL = "http://localhost:7070"
        self.ACCESS_KEY = "8a99c64d820a742083712f3b58b09c2d1fce3a4c3e333c42c8100c4b351d2e9b"
        pass
    
    def fetch_user(self, user, rating_n):
        headers = {
            'Content-Type': 'application/json',
        }
        data = "{{ \"user\": \"{user}\", \"num\": {num} }}".format(
            user=user, num=rating_n)
        try:
            response = requests.post(
                'http://localhost:8000/queries.json', headers=headers, data=data)
            parsed_json = json.loads(response.content)
            item_scores = [str(item) for item_with_score in parsed_json['itemScores']
                        for item in item_with_score.values()]
            # even out rating scores
            line = item_scores
            if len(item_scores) != 2 * rating_n:
                line = line + (2 * rating_n - len(item_scores)) * ['']
            return line
        except (Exception):
            # build a valid structure of eventtime, user_id, error_indice and empty item scores
            return None