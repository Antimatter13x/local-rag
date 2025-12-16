from llama_cpp import Llama

llm = Llama(
    model_path= "models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
    n_ctx=2048
)

print(llm("Say hello in one sentence.")["choices"][0]["text"])
