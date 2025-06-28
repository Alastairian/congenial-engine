import os
import json
from google import genai
from google.genai import types

def handler(request):
    """
    Vercel Python Serverless Function Handler for GenAI Content Generation.

    Expects a JSON POST payload with:
    {
      "user_input": "Your prompt here"
    }

    Returns:
      JSON: { "result": "Generated content..." }
    """
    try:
        if hasattr(request, "json"):
            data = request.json()
        else:
            # If using asgi/wsgi, try parsing body directly
            data = json.loads(request.body.decode("utf-8"))
        user_input = data.get('user_input', '')
    except Exception:
        return {
            "statusCode": 400,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Invalid input. Must be JSON with 'user_input' key."}),
        }

    # Secure API key from environment variable
    api_key = os.environ.get("GOOGLE_GENAI_API_KEY")
    if not api_key:
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "GOOGLE_GENAI_API_KEY not set."}),
        }

    client = genai.Client(api_key=api_key)
    model = "gemini-2.5-pro"
    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=user_input)],
        ),
    ]
    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=-1),
        response_mime_type="text/plain",
    )
    response = ""
    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        response += chunk.text

    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"result": response}),
    }