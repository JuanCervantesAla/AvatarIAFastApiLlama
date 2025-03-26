from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ConnectionManager import ConnectionManager
from LlamaResponse import get_Response

app = FastAPI()

manager = ConnectionManager()

@app.websocket("/comms")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try: 
        while(True):
            data = await websocket.receive_text()
            llama_response = get_Response(data)
            await manager.send_personal_message(f"{llama_response}",websocket)
    except WebSocketDisconnect:
        print("Cliente desconectado.")
    finally:
        await manager.disconnect(websocket)