import requests
from gradio_client import Client

def jsonrpc(payload):
    response = requests.post("http://api.lazybird.app/jsonrpc",
        headers={
            "Content-Type": "application/json",
            "X-Admin-Key": "IAjYjs8l4IsEcrvZj05IKG0WG8eB8opF"
        },
        json=payload
    )

    return response.json()