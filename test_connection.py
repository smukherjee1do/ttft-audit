import os
from openai import OpenAI

client = OpenAI(
       base_url="https://inference.do-ai.run/v1",
       api_key=os.environ["MODEL_ACCESS_KEY"]
   )

response = client.chat.completions.create(
       model="deepseek-v4.1-flash",
       messages=[{"role": "user", "content": "Say hello in one word."}]
   )

print(response.choices[0].message.content)
print(response.usage)