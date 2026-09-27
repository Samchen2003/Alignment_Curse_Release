import os
import json
import requests
from openai import OpenAI


class Target():
    '''
    Temporary implementation of Target class
    '''
    def __init__(self, model):
        self.model = model

    def respond(self, prompt):
        response = self.model.generate("You are a helpful assistant.", prompt, max_length=10000, do_sample=True, temperature=0.6, top_p=0.9)
        return response
    
    
    
class TargetQwen():
    '''
    Temporary implementation of qwen class
    '''    
    def __init__(self, model, api_url, max_tokens):
        self.model = model
        self.api_url = api_url
        self.max_tokens = max_tokens

    def respond(self, prompt):
        api_url = self.api_url
        payload = {
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": prompt}
            ]}
            ],
        }

        r = requests.post(api_url, json=payload)
        json_response = r.json()
        output = json_response["choices"][0]["message"]["content"]
        return output

class TargetGPT():
    def __init__(self, model, max_tokens):
        self.model = model
        self.max_tokens = max_tokens

    def respond(self, prompt):
        
        client = OpenAI()
        
        max_count= 5
        count=0
        while count<max_count:
            count+=1
            try:
                response = client.chat.completions.create(
                    model="gpt-4o-audio-preview-2025-06-03",
                    modalities=["text", "audio"],    
                    audio={"voice": "alloy", "format": "wav"},
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt
                                }
                            ]
                        }
                    ],
                    temperature=0,    
                    max_tokens=self.max_tokens
                )
                output = response.choices[0].message.audio.transcript
                return output
            
            except Exception as e:
                print(f"Try {count+1} failed!!")
        
        
        raise RuntimeError("Unreachable: exhausted retries")    
    
    
class TargetIO():
    '''
    Temporary implementation of qwen class
    '''    
    def __init__(self, model, tokenizer, generation_config):
        self.model = model
        self.tokenizer = tokenizer
        self.generation_config = generation_config

    def respond(self, prompt):
        
        model = self.model
        tokenizer = self.tokenizer
        generation_config = self.generation_config
        
        messages = [
        {
            'role': "user",
            'content': prompt,
        }
        ]
        response = model.chat(tokenizer, generation_config, messages)
        return response
  