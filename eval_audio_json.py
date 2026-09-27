import torch
from transformers import AutoModel, AutoTokenizer
import os
import json, base64
from pathlib import Path
import requests
from openai import OpenAI
from strong_reject.evaluate import evaluate
from prefix import test_prefixes

def get_model_response(audio_path, args):
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    model_name = args.model_name
    if "qwen" in model_name:
        api_url = args.api_url
        payload = {
            "temperature": 0,
            "max_tokens": args.max_tokens,
            "messages": [
                {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}
                ]}
            ],
        }

        r = requests.post(api_url, json=payload)
        json_response = r.json()
        output = json_response["choices"][0]["message"]["content"]

    elif "io" in model_name or "interactive" in model_name:
        model = args.model
        tokenizer = args.tokenizer
        generation_config = args.generation_config
        
        messages = [
            {
                'role': "user",
                'content': [
                    {
                        "type": "audio",
                        "audio": audio_path
                    }
                ]
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
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": b64,      
                                        "format": "wav"
                                    }
                                }
                            ]
                        }
                    ],
                    temperature=0,
                    max_tokens=args.max_tokens
                )
                output = response.choices[0].message.audio.transcript
                return output
            except Exception as e:
                print(f"Try {count+1} failed!!")
                print(e)
        
        raise RuntimeError("Unreachable: exhausted retries")    
        
    else:
        pass
    
    return output


def main(args):
    

    with open(args.text_json, "r") as f:
        text_data = json.load(f)
        
    results=[]
        
    for item in text_data:
        audio_path = item["audio_file"]
        print(f"Processing: {audio_path}")

        response = get_model_response(str(audio_path), args)
        
        jailbreak_label = evaluate(item["jailbreak_prompt"], response, ["strongreject_rubric"])[0]["score"]
        print("Harmful Label: ", jailbreak_label)

        try:
            results.append({
                "goal": item["goal"],
                "jailbreak_prompt": item["jailbreak_prompt"],
                "text_response": item["text_response"],
                "text_score": item["text_score"],
                "audio_response": response,
                "audio_score": jailbreak_label,            
                "audio_file": str(audio_path),           
            })
            
        except:
            results.append({
                "goal": item["goal"],
                "jailbreak_prompt": item["jailbreak_prompt"],
                "audio_response": response,
                "audio_score": jailbreak_label,            
                "audio_file": str(audio_path),           
            })
            

    output_json = args.output_json
    
    os.makedirs(os.path.dirname(output_json), exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("Saved results to:", output_json)
    
    avg_text_score = sum(item.get("audio_score", 0) for item in results) / len(results)
    print("average strongreject score is ", avg_text_score)
    

    count = 0
    total = 0
    for item in results:
        total += 1
        response = item['audio_response']
        if not any([prefix in response for prefix in test_prefixes]):
            count += 1

    print("="*20)
    print(f"Total: {total}, Count: {count}, KeyWords-ASR: {count / total}")
    print("="*20)

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_json", required=True, help="json of textual jailbreaks")
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