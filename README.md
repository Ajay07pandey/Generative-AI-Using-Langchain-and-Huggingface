# Generative-AI-Using-Langchain-and-Huggingface

I have created a Text Generation Chat bot using the Huggingface pretrained models and combine them with Langchain to create a workflow.

For API Calls I Installed the required libraries from HuggingFace

 1. huggingface_hub
 2. transformers
 3. accelerate
 4. bitsandbytes
 5. Langchain


**I have acced the pretrained model using different methods like:**

1. Through Hugging Face Endpoints
2. By creating a LLMChain
3. By creating a LLMpipeLine


## HuggingFaceEndpoint

Used Hugging face Endpoints to access the pre trained models present in Hugging Face

<pre> from langchain_huggingface import HuggingFaceEndpoint
 
 llm = HuggingFaceEndpoint(
 repo_id="mistralai/Mistral-7B-Instruct-v0.3",
 task="text-generation",
 max_new_tokens=100,
 do_sample=False, ) </pre>


**Model Used** - **mistralai/Mistral-7B-Instruct-v0.3**

MistralAI has around 7 Billion parameters so it is not possible to host or download it locally as it is a very large Model hence we have used the API Method to call and access these models. For Text Generation.

## LLMChain: 

<pre> from langchain import PromptTemplate, LLMChain </pre>

LLMChain will Wraps an LLM and a prompt template.
It will help in building logic, memory, and multi-step reasoning.
from prompt template  we have used LLMChain to create a working flow.

## LLM Pipe Line:

<pre> from langchain_huggingface import HuggingFacePipeline </pre>

Also created a LLM pipeline by downloading the simple model **gpt-2**

**AutoModelForCausallLM** will download the model weights and configuration needed for text generation tasks

<pre> from transformers import AutoModelForCausalLM </pre>

**Tokenizer** - Tokenizer converts input text into tokens (numbers) that models can understand, and vice versa.

<pre> from transformers import AutoTokenizer </pre>

This will create a ready to use pipeline object and ready to use generic pipeline utility from transformers

<pre> from transformers import pipeline </pre>

Models can be loaded directly with the **from_model_id** method

The tokenizer is necessary to encode input text into tokens and decode output tokens back to text.

<pre> tokenizer = AutoTokenizer.from_pretrained(model_id) </pre>

As we have to download it locally so we use a lighter model like  **GPT2**

The below pipeline will handle encoding inputs, running the model, and decoding outputs automatically
<pre> pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100
)
</pre>
