import json
import os
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

class ChatGroq:
    """ChatGroq client that sends the prompt to Groq."""
    
    def __init__(self, model, temperature=0, max_retries=2):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.valid_models = [
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]
        
        if model not in self.valid_models:
            raise ValueError(f"Invalid model: {model}")
    
    def invoke(self, messages):
        """Send the prompt to Groq and return the model response."""
        if not isinstance(messages, list) or len(messages) == 0:
            raise ValueError("Messages must be a non-empty list")

        prompt = messages[-1]
        if isinstance(prompt, dict):
            prompt = prompt.get("content", "")
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY environment variable is required")

        payload = json.dumps({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": str(prompt)}],
        }).encode()
        request = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "langchain-course/0.1.0",
            },
            method="POST",
        )

        last_error = None
        for _ in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(request) as response:
                    body = json.loads(response.read().decode())
                content = body["choices"][0]["message"]["content"]
                return MockAIMessage(content)
            except urllib.error.HTTPError as error:
                detail = error.read().decode()
                last_error = Exception(f"Groq request failed ({error.code}): {detail}")
            except urllib.error.URLError as error:
                last_error = Exception(f"Groq request failed: {error.reason}")
        raise last_error

class MockAIMessage:
    """Mock AI message response."""
    def __init__(self, content):
        self.content = content


def implement_set_api_key(api_key):
    """
    IMPLEMENT: Set the GROQ_API_KEY environment variable.
    
    Args:
        api_key (str): Your Groq API key
    """
    # TODO: Your implementation here
    os.environ["GROQ_API_KEY"] = api_key
    return api_key


def check_api_key():
    """
    Check if GROQ_API_KEY is set in environment variables.
    Raise an exception if the API key is not set.
    (This function is provided for you)
    """
    if "GROQ_API_KEY" not in os.environ:
        raise Exception("GROQ_API_KEY environment variable is required")
    return os.environ.get("GROQ_API_KEY")


def implement_llama_3_3_model():
    """
    IMPLEMENT: Create and return a ChatGroq instance.
    Use a model id this Groq account can access.
    Set temperature=0 for consistent responses.
    """
    return ChatGroq(model="openai/gpt-oss-120b", temperature=0)


def implement_query_model(model, prompt):
    """
    IMPLEMENT: Send a query to the model and return the response content.
    
    Args:
        model: The ChatGroq model instance
        prompt: The text prompt to send
        
    Returns:
        str: The response content
    """
    response = model.invoke([prompt])
    return response.content


def implement_compare_models(prompt):
    """
    IMPLEMENT: Query the Llama 3.3 model and return its response.
    
    Args:
        prompt: The text prompt to send
        
    Returns:
        dict: Dictionary with the Llama 3.3 response
    """
    llama33 = implement_llama_3_3_model()
    return {
        "openai/gpt-oss-120b": implement_query_model(llama33, prompt),
    }


def main():
    """
    Main function to test your implementations.
    """
    print("🚀 Groq Model Switching Exercise (LangChain Integration)")
    print("=" * 55)
    print("📝 This exercise simulates langchain-groq package behavior!")
    print("🌐 Model names should match console.groq.com exactly")
    print()
    
    try:
        # Test your set_api_key implementation
        print("🔑 Setting API key...")
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY environment variable is required")
        implement_set_api_key(api_key)
        
        # Check if API key was set correctly
        check_api_key()
        print("✓ API key validation working!")
        
        # Test prompt
        test_prompt = "List 5 cool things about programming."
        
        # Test your model implementation
        print(f"\n🤖 Testing your model implementation:")
        llama33 = implement_llama_3_3_model()
        response33 = implement_query_model(llama33, test_prompt)
        print(f"Response: {response33}\n")
        
        # Test your comparison implementation
        print("🔄 Testing your model comparison:")
        comparison = implement_compare_models(test_prompt)
        print("Comparison results:")
        for model, response in comparison.items():
            print(f"  {model}: {response}")
        
        print("\n🎉 All implementations working!")
        print("✅ Great job implementing the LangChain-Groq patterns!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        if "GROQ_API_KEY" in str(e):
            print("\n💡 Check your implement_set_api_key() function!")
        else:
            print("📝 Check your function implementations!")
            print("🌐 Verify model names match console.groq.com exactly")


if __name__ == "__main__":
    main()