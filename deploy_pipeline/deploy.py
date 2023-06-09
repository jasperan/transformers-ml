from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from transformers import pipeline
import asyncio



async def homepage(request):
    """
    A coroutine that handles incoming requests to the homepage.

    Args:
        request (Request): The incoming request object.

    Returns:
        JSONResponse: A JSON response containing the output of the request.
    """
    # Read the request body and decode it to a string.
    payload = await request.body()
    string = payload.decode("utf-8")

    # Create a queue to hold the response.
    response_q = asyncio.Queue()

    # Add the request and response queue to the model queue.
    await request.app.model_queue.put((string, response_q))

    # Wait for the response to be added to the response queue.
    output = await response_q.get()

    # Return the output as a JSON response.
    return JSONResponse(output)



async def server_loop(q):
    """
    This function listens for incoming messages in a queue, processes them using a BERT model,
    and puts the results into a response queue.

    :param q: The queue to listen for incoming messages.
    """
    # Initialize the BERT pipeline model
    pipe = pipeline(model="bert-base-uncased")

    while True:
        # Get the next message from the input queue
        (string, response_q) = await q.get()

        # Process the message using the BERT pipeline model
        out = pipe(string)

        # Put the result into the response queue
        await response_q.put(out)


app = Starlette(
    routes=[
        Route("/", homepage, methods=["POST"]),
    ],
)


@app.on_event("startup")
async def startup_event() -> None:
    """
    This function is called when the FastAPI application starts up. It creates a queue and assigns it to the app's model_queue attribute. It also starts a server_loop task.
    """
    q = asyncio.Queue()
    app.model_queue = q
    asyncio.create_task(server_loop(q))