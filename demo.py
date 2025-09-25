import requests
import json

response = requests.post(
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={
        "Authorization": "Bearer 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",
        "Content-Type": "application/json",
       
    },
    data=json.dumps({
        "model": "x-ai/grok-4-fast:free",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this image?"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                        }
                    }
                ]
            }
        ],
    })
)

if response.status_code == 200:
    print(response.json()["choices"][0]["message"]["content"])
else:
    print(f"Error: {response.status_code}, {response.text}")