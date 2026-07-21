from llama_cpp import Llama


llm = Llama(
    model_path="models/qwen/qwen2.5-3b-instruct-q4_k_m.gguf",
    n_ctx=4096,
    n_threads=6
)


response = llm(
    "When was the Battle of Little Bighorn fought?",
    max_tokens=100,
    temperature=0
)


print(response["choices"][0]["text"])