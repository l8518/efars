import requests
import datetime
import time
from app.evaluation.run.abstract_provider_adapter import AbstractProviderAdapter


class BasicRecommenderProviderAdapter(AbstractProviderAdapter):
    def __init__(self):
        self.URL = "http://localhost:7070"
        self.ACCESS_KEY = "8a99c64d820a742083712f3b58b09c2d1fce3a4c3e333c42c8100c4b351d2e9b"
        pass
    
    def emit_rating(self, user_id, item_id, rating, retry=0):
        try:
            headers = {
                'Content-Type': 'application/json',
            }

            params = (
                ('accessKey', '8a99c64d820a742083712f3b58b09c2d1fce3a4c3e333c42c8100c4b351d2e9b'),
            )

            eventTime = str(datetime.datetime.utcnow().replace(
                tzinfo=datetime.timezone.utc).isoformat())
            data = "{{\n  \"event\" : \"rate\",\n  \"entityType\" : \"user\",\n  \"entityId\" : \"{uid}\",\n  \"targetEntityType\" : \"item\",\n  \"targetEntityId\" : \"{target_id}\",\n  \"eventTime\" : \"{eventTime}\",\n  \"properties\" : {{ \"rating\" : {rating} }}\n }}".format(
                uid=user_id, target_id=item_id, eventTime=eventTime, rating=rating
            )
            response = requests.post(
                'http://localhost:7070/events.json', headers=headers, params=params, data=data)
            return (0, retry)
        except (Exception):
            if retry < 3:
                time.sleep(0.5)
                return self.emit_rating(user_id, item_id, rating, retry=(retry+1))
            else:
                return (1, retry)
