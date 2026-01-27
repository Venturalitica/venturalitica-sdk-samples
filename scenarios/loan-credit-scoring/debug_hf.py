def test_gguf_chat_local(repo_id="BSC-LT/ALIA-40b-instruct-2512-GGUF", filename="ALIA-40b-instruct-2512-Q8_0.gguf"):
    print(f"\n--- Testing GGUF CHAT LOCAL (ChatLlamaCpp): {repo_id}/{filename} ---")
    try:
        from huggingface_hub import hf_hub_download
        from langchain_community.chat_models import ChatLlamaCpp
        from langchain_core.messages import HumanMessage
        
        print(f"   Downloading/Loading model using ChatLlamaCpp...")
        model_path = hf_hub_download(repo_id=repo_id, filename=filename)

        llm = ChatLlamaCpp(
            model_path=model_path,
            n_ctx=2048,
            n_gpu_layers=-1, # Use all GPU layers if possible
        )
        
        print(f"   Inference starting...")
        messages = [
            HumanMessage(content="¿Quién eres y cuál es tu propósito principal?")
        ]
        response = llm.invoke(messages)
        print(f"✅ Success! Response: {response.content}")
    except Exception as e:
        print(f"❌ Failed:")
        print(f"   Type: {type(e).__name__}")
        print(f"   Message: {str(e)}")

if __name__ == "__main__":
    print(f"\n🚀 Starting GGUF ALIA (ChatLlamaCpp) Debug Session\n")
    test_gguf_chat_local()
