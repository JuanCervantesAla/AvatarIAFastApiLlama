from fastapi import WebSocket

class ConnectionManager:
    """Class with sockets events"""
    def __init__(self):
        self.active_connections =[] #Keep track of the connections with an array
    
    async def connect(self,websocket: WebSocket):#Connection event
        await websocket.accept()#Accept connection
        self.active_connections.append(websocket)#Append to active connections
    
    async def send_personal_message(self, message: str, websocket:WebSocket):
        await websocket.send_text(message)#Direct message
    
    def disconnect(self, websocket:WebSocket):#If gets a discconection from the socket 
        self.active_connections.remove(websocket)#Remove from the connections