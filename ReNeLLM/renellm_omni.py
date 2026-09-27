import torch
from transformers import AutoTokenizer, AutoModel
import time
import json
import os
import argparse
import random
import requests, json

from utils.llm_completion_utils import chatCompletion, claudeCompletion
from utils.prompt_rewrite_utils import shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange
from utils.scenario_nest_utils import SCENARIOS
from utils.harmful_classification_utils import harmful_classification

from prefix import test_prefixes
from strong_reject.evaluate import evaluate


def qwen_inference(prompt, args):
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



def interactive_inference(prompt, args, model=None, tokenizer=None, generation_config=None):
    if model is None or tokenizer is None or generation_config is None:
        raise ValueError("interactive_inference requires model, tokenizer, and generation_config")
    messages = [{"role": "user", "content": prompt}]
    return model.chat(tokenizer, generation_config, messages)

def main(args):

    with open(args.data_path, "r") as f:
        data = json.load(f)
    
        
    if "io" in args.attack_model or "interactive" in args.attack_model:

        print("===========interactive started===============\n")
        path = args.api_url
        model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True).eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=True)

        print("model loaded!!!")

        generation_config = dict(max_new_tokens=args.max_tokens, do_sample=False)

    operations = [shortenSentence, misrewriteSentence, changeOrder, addChar, languageMix, styleChange]
    scenarios = SCENARIOS

    data_list = []

    for idx, item in enumerate(data):

        start_time = time.time()
        
        harm_behavior = item["goal"]
        
        temp_harm_behavior = harm_behavior
        
        

        if not os.path.exists('results/temp'):
            os.makedirs('results/temp')

        if idx != 0 and idx % 10 == 0:
            file_name = f"results/temp/{args.save_suffix}_{idx}.json"

            with open(file_name, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)
            file_path = os.path.abspath(file_name)
            print(f"\nThe temporary file has been saved to: {file_path}\n")

        loop_count = 0
        rewrite_fail = 0
        
        while True:
            print(
            "\n################################\n"
            f"Current Data: {idx+1}/{len(data)}, {harm_behavior}\n"
            f"Current Iteration Round: {loop_count+1}/{args.iter_max}\n"
            "################################\n")
            
            rewrite_count = 0
            
            while True:
                if rewrite_count >= args.rewrite_max:
                    print("!!! Rewrite maximum reached !!!")
                    rewrite_fail = 1
                    break
                
                rewrite_count += 1
                print(f"******* Start idx {idx} Prompt Rewriting! *******")
                n = random.randint(1, 6)
                operation_indexes = random.sample(range(6), n)
                print(f"The number of rewriting functions is: {n}")
                print(f"The order of the rewriting operations is: {operation_indexes}\n")

                temp_rewrite_results = [['original prompt', temp_harm_behavior]]
                for index in operation_indexes:
                    print(f"Excute function {index}: {operations[index].__name__}")
                    harm_behavior = operations[index](args, harm_behavior)
                    print(f"Rewritten prompt: {harm_behavior}\n")
                    temp_rewrite_results.append([operations[index].__name__, harm_behavior])
                    
                print("******* Check if the prompt has been modified to be harmless *******")
                label = harmful_classification(args, harm_behavior)
                print(f"\nThe label given by {args.judge_model} is: {label}\n")
                if label == "1":
                    print(f"******* Prompt idx {idx} Rewriting Done! *******\n")
                    break
                else:
                    print(f"******* Prompt idx {idx} Rewriting Failed! *******\n")
                    harm_behavior = temp_harm_behavior
                    continue



            
            print(f"******* Start idx {idx} Prompt Scenario Nesting! *******")
            print(f"harm_behavior:\n{harm_behavior}\n")
            scenario = random.choice(scenarios)
            nested_prompt = scenario.replace("<>", harm_behavior)
            print(f"nested_prompt:\n{nested_prompt}\n")

            print(f"******* Start idx {idx} Prompt Jailbreaking on {args.attack_model}! *******")
            
            if "gpt" in args.attack_model:
                messages = [{"role": "user", "content": nested_prompt}]
                
                attack_output = chatCompletion(
                    args.attack_model,
                    messages,
                    args.temperature,
                    args.retry_times,
                    args.round_sleep,
                    args.fail_sleep,
                    args.gpt_api_key,
                    args.gpt_base_url
                )
            
            elif "qwen" in args.attack_model:
                attack_output = qwen_inference(nested_prompt, args)
                
            elif "io" in args.attack_model:
                attack_output = interactive_inference(nested_prompt, args, model, tokenizer, generation_config)   
                
            else:
                attack_output = claudeCompletion(
                    args.attack_model,
                    args.max_tokens,
                    args.temperature,
                    nested_prompt,
                    args.retry_times,
                    args.round_sleep,
                    args.fail_sleep,
                    args.claude_api_key,
                    args.claude_base_url
                )
                
            print(f"The output of the attacked model {args.attack_model} is:\n\n{attack_output}\n")

            loop_count += 1
            
            if rewrite_fail == 1:
                print(f"\n******* Exceeded the maximum number of rewrite iterations {args.rewrite_max}, adopt the current round results and end the loop.*******\n")
                
                sr_label = evaluate(nested_prompt, attack_output, ["strongreject_rubric"])[0]["score"]
                item = {}
                item['goal']  = temp_harm_behavior
                item['rewritten_prompt']  = harm_behavior
                item['jailbreak_prompt']  = nested_prompt
                item['text_response']  = attack_output
                item['text_score'] = sr_label
                item['iteration_count']  = loop_count
                data_list.append(item)

                break


            elif "apologize" not in attack_output and "sorry" not in attack_output.lower():

                print(f"******* Determine if the attacked model's response to the nested idx {idx} prompt is harmful *******")
                jailbreak_label = harmful_classification(args, attack_output)
            
            else:
                if loop_count < args.iter_max:
                    print(f"\nJailbreaking idx {idx} Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                    sr_label = evaluate(nested_prompt, attack_output, ["strongreject_rubric"])[0]["score"]
                    item = {}
                    item['goal']  = temp_harm_behavior
                    item['rewritten_prompt']  = harm_behavior
                    item['jailbreak_prompt']  = nested_prompt
                    item['text_response']  = attack_output
                    item['text_score'] = sr_label
                    item['iteration_count']  = loop_count
                    data_list.append(item)
                    

                    break

            if jailbreak_label == "1":
                print(f"\n******* Jailbreaking idx {idx} Prompt Successful! *******\n")
                sr_label = evaluate(nested_prompt, attack_output, ["strongreject_rubric"])[0]["score"]
                item = {}
                item['goal']  = temp_harm_behavior
                item['rewritten_prompt']  = harm_behavior
                item['jailbreak_prompt']  = nested_prompt
                item['text_response']  = attack_output
                item['text_score'] = sr_label
                item['iteration_count']  = loop_count
                 

                end_time = time.time()  
                elapsed_time = end_time - start_time  
                item['time_cost'] = elapsed_time

                data_list.append(item)

                break
            else:
                if loop_count < args.iter_max:
                    print(f"\nJailbreaking idx {idx} Prompt Failed!\n")
                    harm_behavior = temp_harm_behavior
                    continue
                else:
                    print(f"\n******* Exceeded the maximum number of iterations {args.iter_max}, adopt the current round results and end the loop.*******\n")
                    sr_label = evaluate(nested_prompt, attack_output, ["strongreject_rubric"])[0]["score"]
                    item = {}
                    item['goal']  = temp_harm_behavior
                    item['rewritten_prompt']  = harm_behavior
                    item['jailbreak_prompt']  = nested_prompt
                    item['text_response']  = attack_output
                    item['text_score'] = sr_label
                    item['iteration_count']  = loop_count
                    data_list.append(item)

                    break

    file_name = f"results/renellm-{args.attack_model}.json"

    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(data_list, f, ensure_ascii=False, indent=4)

    file_path = os.path.abspath(file_name)
    print(f"\nThe final file has been saved to:\n{file_path}\n")
    
    
    avg_text_score = sum(item.get("text_score", 0) for item in data_list) / len(data_list)
    print("average strongreject score is ", avg_text_score)
    


    count = 0
    total = 0
    for item in data_list:
        total += 1
        response = item['text_response']
        if not any([prefix in response for prefix in test_prefixes]):
            count += 1

    print("="*20)
    print(f"Total: {total}, Count: {count}, KeyWords-ASR: {count / total}")
    print("="*20)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/advbench.json')
    parser.add_argument('--prompt', type=str, default=None)
    parser.add_argument('--rewrite_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for rewriting the prompt')
    parser.add_argument('--judge_model', type=str, default="gpt-3.5-turbo", choices=["gpt-3.5-turbo", "gpt-4"], help='model uesd for harmful classification')
    parser.add_argument('--attack_model', type=str, default="qwen2.5-omni-7B", 
                        help='model to be attacked (victim model)')
    parser.add_argument('--api_url', type=str, help='vllm service; pass in interactiveomni model checkpoint here when using this model.')
    parser.add_argument('--iter_max', type=int, default=20, help='max iteration times')
    parser.add_argument('--rewrite_max', type=int, default=10, help='max rewrite times')
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument('--temperature', type=float, default=0, help='model temperature')
    parser.add_argument('--round_sleep', type=int, default=1, help='sleep time between every round')
    parser.add_argument('--fail_sleep', type=int, default=1, help='sleep time for fail response')
    parser.add_argument('--retry_times', type=int, default=10, help='retry times when exception occurs')
    parser.add_argument('--save_suffix', type=str, default='normal')
    parser.add_argument("--gpt_api_key", type=str, default=None)
    parser.add_argument("--gpt_base_url", type=str, default=None)

    args = parser.parse_args()
    
    if args.gpt_api_key is None:
        args.gpt_api_key = os.environ["OPENAI_API_KEY"]

    main(args)
