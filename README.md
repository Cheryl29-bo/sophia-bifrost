# sophia-bifrost

## Deploy with Docker

The Docker image runs Sophia on the platform-provided `PORT` (default `8000`)
and runs the Bifrost gateway privately on `127.0.0.1:8080`.

Build and run locally:

```sh
docker build -t sophia-bifrost .
docker run --rm -p 8000:8000 \
  -e GEMINI_API_KEY=your-gemini-api-key \
  -e TAVILY_API_KEY=your-tavily-api-key \
  sophia-bifrost
```

For Render or Railway, deploy from this Dockerfile and add `GEMINI_API_KEY`
and `TAVILY_API_KEY` as service environment variables. Do not commit `.env`;
it is excluded from Git and the Docker build context. Bifrost's generated
configuration refers to the Gemini key through the environment and does not
write its value to the image. Startup uses `gemini/gemini-3.8-flash-lite` as
the Bifrost request fallback; set `BIFROST_FALLBACK_MODEL` to override it.