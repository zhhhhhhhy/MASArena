import argparse
import contextlib
import sys
import os
import datetime
from dotenv import load_dotenv
from tqdm import tqdm
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from openai import AzureOpenAI, OpenAI

from lib.utils import (
    all_at_once as gpt_all_at_once,
    step_by_step as gpt_step_by_step,
    binary_search as gpt_binary_search
)

from lib.local_model import (
    analyze_all_at_once_local,
    analyze_step_by_step_local,
    analyze_binary_search_local
)


KNOWN_GPT_MODELS = {"gpt-4o", "gpt-4", "gpt-4o-mini"}
LOCAL_LLAMA_ALIASES = {"llama-8b", "llama-70b"}
LOCAL_QWEN_ALIASES = {"qwen-7b", "qwen-72b"}
LOCAL_MODEL_ALIASES = LOCAL_LLAMA_ALIASES | LOCAL_QWEN_ALIASES
ALL_MODELS = list(KNOWN_GPT_MODELS | LOCAL_MODEL_ALIASES)

LOCAL_MODEL_MAP = {
    "llama-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-70b": "meta-llama/Llama-3.1-70B-Instruct",
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen-72b": "Qwen/Qwen2.5-72B-Instruct",
}

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze multi-agent responses for failure attribution.")

    parser.add_argument(
        "--method",
        type=str,
        required=True,
        choices=["all_at_once", "step_by_step", "binary_search"],
        help="The analysis method to use."
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=ALL_MODELS,
        help=f"Model identifier. Choose from: {', '.join(ALL_MODELS)}"
    )
    parser.add_argument(
        "--directory_path",
        type=str,
        default="../results/agent_responses",
        help="Path to the directory containing agent response JSON files. Default: '../results/agent_responses'."
    )

    parser.add_argument(
        "--api_key", type=str, default="",
        help="OpenAI API Key. Uses OPENAI_API_KEY or AZURE_OPENAI_API_KEY env var if available."
    )
    parser.add_argument(
        "--azure_endpoint", type=str, default="",
        help="Azure OpenAI Endpoint URL. If not provided, will use standard OpenAI API. Uses AZURE_OPENAI_ENDPOINT env var if available."
    )
    parser.add_argument(
        "--openai_base_url", type=str, default="",
        help="OpenAI API Base URL. Uses OPENAI_API_BASE env var if available."
    )
    parser.add_argument(
        "--api_version", type=str, default="2024-08-01-preview",
        help="Azure OpenAI API Version. Used only for GPT models."
    )
    parser.add_argument(
        "--max_tokens", type=int, default=1024,
        help="Maximum number of tokens for GPT API response. Used only for GPT models."
    )

    parser.add_argument(
        "--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for local model inference (e.g., 'cuda', 'cuda:0', 'cpu'). Default: 'cuda' if available, else 'cpu'."
    )

    args = parser.parse_args()

    # Get API credentials from environment if not provided
    if not args.api_key:
        args.api_key = os.getenv('OPENAI_API_KEY', '') or os.getenv('AZURE_OPENAI_API_KEY', '')
    if not args.azure_endpoint:
        args.azure_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '')
    if not args.openai_base_url:
        args.openai_base_url = os.getenv('OPENAI_API_BASE', '')

    client_or_model_obj = None
    model_type = None  # gpt, llama, qwen
    model_family = None 
    model_id_or_deployment = args.model

    if args.model in KNOWN_GPT_MODELS:
        model_type = 'gpt'
        model_family = 'gpt'
        print(f"Selected GPT model: {args.model}")
       
        if not args.api_key:
            print("Error: --api_key or OPENAI_API_KEY/AZURE_OPENAI_API_KEY environment variable is required for GPT models")
            sys.exit(1)
        
        try:
            if args.azure_endpoint:
                # Use Azure OpenAI
                client_or_model_obj = AzureOpenAI(
                    api_key=args.api_key,
                    api_version=args.api_version,
                    azure_endpoint=args.azure_endpoint,
                )
                print(f"Successfully initialized AzureOpenAI client for endpoint: {args.azure_endpoint}")
            else:
                # Use standard OpenAI API
                openai_kwargs = {"api_key": args.api_key}
                if args.openai_base_url:
                    openai_kwargs["base_url"] = args.openai_base_url
                    print(f"Using custom OpenAI base URL: {args.openai_base_url}")
                
                client_or_model_obj = OpenAI(**openai_kwargs)
                print(f"Successfully initialized OpenAI client")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            sys.exit(1)

    elif args.model in LOCAL_MODEL_ALIASES:
        model_type = 'local'
        model_id_or_deployment = LOCAL_MODEL_MAP[args.model]

        if args.model in LOCAL_LLAMA_ALIASES:
            model_family = 'llama'
            print(f"Selected local Llama model: {args.model} ({model_id_or_deployment}) on device {args.device}")
            if not pipeline:
                 print("Error: transformers library not found or pipeline could not be imported.")
                 sys.exit(1)
            try:
                 print(f"Initializing Llama pipeline for {model_id_or_deployment}...")
                 client_or_model_obj = pipeline(
                     "text-generation",
                     model=model_id_or_deployment,
                     model_kwargs={"torch_dtype": torch.bfloat16},
                     device=args.device,
                 )
                 print(f"Successfully initialized Llama pipeline on {args.device}.")
            except Exception as e:
                print(f"Error initializing Llama pipeline for {model_id_or_deployment}: {e}")
                sys.exit(1)

        elif args.model in LOCAL_QWEN_ALIASES:
            model_family = 'qwen'
            print(f"Selected local Qwen model: {args.model} ({model_id_or_deployment}) on device {args.device}")
            if not AutoModelForCausalLM or not AutoTokenizer:
                 print("Error: transformers library not found or specific classes could not be imported.")
                 sys.exit(1)
            try:
                 print(f"Initializing Qwen model and tokenizer for {model_id_or_deployment}...")
                 qwen_model = AutoModelForCausalLM.from_pretrained(
                    model_id_or_deployment,
                    torch_dtype="auto",
                    device_map=args.device
                 )
                 qwen_tokenizer = AutoTokenizer.from_pretrained(model_id_or_deployment)
                 client_or_model_obj = (qwen_model, qwen_tokenizer)
                 print(f"Successfully initialized Qwen model and tokenizer on {args.device}.")
            except Exception as e:
                print(f"Error initializing Qwen model/tokenizer for {model_id_or_deployment}: {e}")
                print("Make sure you have sufficient VRAM/RAM and necessary libraries (transformers, torch, accelerate).")
                sys.exit(1)
    else:
        print(f"Error: Invalid model '{args.model}' specified.")
        sys.exit(1)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{args.method}_{args.model.replace('/','_')}_agent_responses.txt"
    output_filepath = os.path.join(output_dir, output_filename)

    print(f"Analysis method: {args.method}")
    print(f"Model Alias: {args.model} (Family: {model_family})")
    print(f"Output will be saved to: {output_filepath}")

    try:
        with open(output_filepath, 'w', encoding='utf-8') as output_file, contextlib.redirect_stdout(output_file):
            print(f"--- Starting Analysis: {args.method} ---")
            print(f"Timestamp: {datetime.datetime.now()}")
            print(f"Model Family: {model_family}")
            print(f"Model Used: {model_id_or_deployment}")
            print(f"Input Directory: {args.directory_path}")
            print("-" * 20)

            if model_type == 'gpt':
                if args.method == "all_at_once":
                    gpt_all_at_once(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "step_by_step":
                    gpt_step_by_step(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
                elif args.method == "binary_search":
                    gpt_binary_search(
                        client=client_or_model_obj,
                        directory_path=args.directory_path,
                        model=args.model,
                        max_tokens=args.max_tokens
                    )
            elif model_type == 'local':
                if args.method == "all_at_once":
                    analyze_all_at_once_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        model_family=model_family
                    )
                elif args.method == "step_by_step":
                    analyze_step_by_step_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        model_family=model_family
                    )
                elif args.method == "binary_search":
                    analyze_binary_search_local(
                        model_obj=client_or_model_obj,
                        directory_path=args.directory_path,
                        model_family=model_family
                    )

            else:
                 print(f"Internal Error: Unknown model_type '{model_type}' during function call.")

            print("-" * 20)
            print(f"--- Analysis Complete ---")

        print(f"Analysis finished. Output saved to {output_filepath}")

    except Exception as e:
        print(f"\n!!! An error occurred during analysis or file writing: {e} !!!", file=sys.stderr)
  
if __name__ == "__main__":
    main()