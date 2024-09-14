import logging
import vllm
from typing import Dict, List, Union
from flask import Flask, request
from flask_cors import CORS
from waitress import serve
import concurrent.futures
from transformers import set_seed

set_seed(0)

# Initialize Flask
app = Flask(__name__)
cors = CORS(app)

# Initialize Large Language Model
llm_checkpoint_path = "/mnt/sdf/OTT-QAMountSpace/ModelCheckpoints/Ours/llm/meta-llama/Meta-Llama-3.1-8B-Instruct"
tensor_parallel_size = 1
gpu_memory_utilization = 0.4
max_model_length = 6400

llm  = vllm.LLM(
            llm_checkpoint_path,
            worker_use_ray = True,
            tensor_parallel_size = tensor_parallel_size, 
            gpu_memory_utilization = gpu_memory_utilization, 
            trust_remote_code = True,
            dtype = "half", # note: bfloat16 is not supported on nvidia-T4 GPUs
            max_model_len = max_model_length, # input length + output length
            enforce_eager = True,
            seed = 0,
        )
tokenizer = llm.get_tokenizer()

@app.route("/generate", methods=["GET", "POST", "OPTIONS"])
def generate():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()
    
    prompt_list = params["prompt_list"]
    max_tokens = params.get("max_tokens", 64)
    
    responses = llm.generate(
                                prompt_list,
                                vllm.SamplingParams(
                                    seed = 0,
                                    n = 1,  # Number of output sequences to return for each prompt.
                                    top_p = 0.9,  # Float that controls the cumulative probability of the top tokens to consider.
                                    temperature = 0,  # randomness of the sampling
                                    skip_special_tokens = True,  # Whether to skip special tokens in the output.
                                    max_tokens = max_tokens,  # Maximum number of tokens to generate per output sequence.
                                    logprobs = 1
                                ),
                                use_tqdm = False
                            )
    response_list = []
    for response in responses:
        response_list.append(response.outputs[0].text)
    
    response = {"response_list": response_list}
    
    return response

@app.route("/trim", methods=["GET", "POST", "OPTIONS"])
def trim():
    # Handle GET and POST requests
    if request.method == "POST":
        params: Dict = request.json
    else:
        params: Dict = request.args.to_dict()
    
    raw_text = params["raw_text"]
    trim_length = params["trim_length"]
    
    tokenized_text = tokenizer.encode(raw_text)
    trimmed_tokenized_text = tokenized_text[ : trim_length]
    trimmed_text = tokenizer.decode(trimmed_tokenized_text)
    
    response = {"trimmed_text": trimmed_text}
    
    return response

if __name__ == "__main__":
    logging.basicConfig(
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
        datefmt="%m/%d %H:%M:%S",
        level=logging.INFO,
    )
    serve(app, host="0.0.0.0", port=5005)