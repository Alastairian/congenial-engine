# IAI-IPS Twine Cognition GenAI API

A Vercel-ready Python serverless function wrapping the Google GenAI client for content generation.

## Usage

Deploy to Vercel. POST a JSON payload to `/api/generate`:

```json
{
  "user_input": "Your prompt here"
}
```

Response:

```json
{
  "result": "Generated content..."
}
```

## Environment Variables

- `GOOGLE_GENAI_API_KEY`  
  Set this in your Vercel project environment variables.

## Local Development

```bash
pip install -r requirements.txt
export GOOGLE_GENAI_API_KEY=your-key
# Run with your preferred Python HTTP server (see Vercel docs for local python API testing)
```

## Vercel Configuration

- Python serverless function in `api/generate.py`
- Dependencies in `requirements.txt`
- `vercel.json` ensures Python 3.11 runtime for the API endpoint.

## License

See repository license.