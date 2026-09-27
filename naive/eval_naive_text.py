import torch
from transformers import AutoModel, AutoTokenizer
import os
import json, base64
from pathlib import Path
import requests
import argparse
from openai import OpenAI
from strong_reject.evaluate import evaluate
from prefix import test_prefixes


def get_model_response(prompt, args, model=None, tokenizer=None, generation_config=None):
    model_name = args.model_name
        
    if "qwen" in model_name:
        api_url = args.api_url
        payload = {
            "temperature": 0,
            "max_tokens": args.max_tokens,
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
    
    
    
    elif "io" in model_name or "interactive" in model_name:
        model = args.model
        tokenizer = args.tokenizer
        generation_config = args.generation_config
        
        messages = [
        {
            'role': "user",
            'content': prompt,
        }
        ]
        response = model.chat(tokenizer, generation_config, messages)
        return response

        
    elif "gpt" in model_name:
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
                    max_tokens=args.max_tokens,
                )
                output = response.choices[0].message.audio.transcript
                return output
            
            except Exception as e:
                print(f"Try {count+1} failed!!")
        
        
        raise RuntimeError("Unreachable: exhausted retries")    
    
        
    else:
        pass
    
    return output


def main(args):
    
    input_json = args.input_json
    output_json = args.output_json
    print(f"!!!!Eval {args.model_name}")
    
    
    with open(input_json, 'r') as f:
        input = json.load(f)

    results = []

    for item in input:
        print(f"Current: {item['goal']}")
        response = get_model_response(item["goal"], args)

            
        jailbreak_label = evaluate(item["goal"], response, ["strongreject_rubric"])[0]["score"]
        print("Harmful Label: ", jailbreak_label)

        results.append({
            "goal": item["goal"],
            "jailbreak_prompt": item["goal"],
            "text_response": response,
            "text_score": jailbreak_label,
        })

    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to:", output_json)
    
    avg_text_score = sum(item.get("text_score", 0) for item in results) / len(results)
    print("average strongreject score is ", avg_text_score)
   
    

    count = 0
    total = 0
    for item in results:
        total += 1
        response = item['text_response']
        if not any([prefix in response for prefix in test_prefixes]):
            count += 1

    print("="*20)
    print(f"Total: {total}, Count: {count}, KeyWords-ASR: {count / total}")
    print("="*20)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", default="data/advbench.json")
    parser.add_argument("--output_json", required=True, help="Output JSON path")
    parser.add_argument("--max_tokens", default=512, help="max tokens for llms")
    parser.add_argument('--api_url', type=str, help='vllm service')
    parser.add_argument('--model_name', type=str, default="qwen2.5-omni-7B", 
                    help='model to be attacked (victim model)')
    args = parser.parse_args()
    
    if "interactive" in args.model_name or "io" in args.model_name:
        print("===========interactive started===============\n")
        path = args.api_url
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)

        print("model loaded!!!")

        generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)
        
        args.model = model
        args.tokenizer = tokenizer
        args.generation_config = generation_config


    main(args)