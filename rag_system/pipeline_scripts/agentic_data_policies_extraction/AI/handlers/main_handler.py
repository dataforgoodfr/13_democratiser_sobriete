from together import Together

def get_client():
    return Together(api_key='71b3fef71bec9ec08ff91058e4028e1cf1e971dae9b2bfbb6e9c013c2401411f')

def get_response(client, prompt_conclusion):
    response = client.chat.completions.create(
        # model="deepseek-ai/DeepSeek-V3", ### pour un modèle payant
        # model="deepseek-ai/DeepSeek-R1-0528", ### The maximum rate limit for this model is 0.3 queries and 60000 tokens per minute.
        model="meta-llama/Llama-Vision-Free",

        messages=[
            {
                "role": "system",
                "content": "You are an information extraction assistant."
            },
            {
                "role": "user",
                "content": prompt_conclusion
            }
        ]
    )

    return response.choices[0].message.content