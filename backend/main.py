from researcher import graph
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}


from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from researcher import graph
import json
import asyncio

app = FastAPI()


@app.post("/research")
async def research_stream(request: Request):
    # Get the JSON data from the request body
    inputs = await request.json()

    # Define the keys that should have default values
    default_keys = ["target_market", "competitors", "pricing_strategy", "context"]

    # Set default values for missing keys
    for key in default_keys:
        if key not in inputs or not inputs[key]:
            inputs[key] = " "

    async def generate():
        # Process the inputs
        for event in graph.stream(inputs, stream_mode="values"):
            research = event
            # print(event)  # This will print to your server console

            # If we have a research result, yield it as a stream
            if "research_result" in research:
                # Assuming research_result is a markdown string
                markdown_content = research["research_result"]
                # Split the markdown content into smaller chunks
                chunk_size = 32  # Adjust this value as needed
                for i in range(0, len(markdown_content), chunk_size):
                    chunk = markdown_content[i : i + chunk_size]
                    yield f"data: {json.dumps({'markdown_chunk': chunk})}\n\n"
                    await asyncio.sleep(0.1)  # Small delay to prevent flooding

        # Signal the end of the stream
        yield 'data: {"status": "complete"}\n\n'

    return StreamingResponse(generate(), media_type="text/event-stream")
