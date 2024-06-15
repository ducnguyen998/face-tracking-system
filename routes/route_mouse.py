import sys
import json
import threading
import random as rd
import queue
import asyncio

from fastapi import FastAPI, WebSocket
from fastapi.websockets import WebSocketDisconnect
from typing import List


from pathlib import (
    Path
)

from fastapi import (
    APIRouter,
    Response,
    Depends,
    Request
)

from sse_starlette.sse import (
    EventSourceResponse
)

#from face_tracking import (
#    main,
#)
#
#from face_tracking.main import (
#    MouseObject
#)

#import main
from main_2 import (
    MouseObject,
    WINDOW_SIZE_WIDTH,
    run,
    test
)

##
MOUSE_OBJECT = MouseObject(0,0,0)



def mouse_callback(mouse_obj):
    global MOUSE_OBJECT 
    MOUSE_OBJECT = mouse_obj
    #print(f'Mouse callback : x={MOUSE_OBJECT.x}, y={MOUSE_OBJECT.y}, s={MOUSE_OBJECT.s}')
##

thread = threading.Thread(target=run, args=(mouse_callback,))
thread.start()

sys.path.append(str(Path(__file__).parent))

router = APIRouter()

@router.post("/report")
async def mouse_report():
    global MOUSE_OBJECT
    mouse_obj = MOUSE_OBJECT
    ret = {
        'state' : int(mouse_obj.s),
        'x' : WINDOW_SIZE_WIDTH - int(mouse_obj.x),
        'y' : int(mouse_obj.y)
    }
    print(ret)
    return Response(content=json.dumps(ret), media_type='content/json')



class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    global MOUSE_OBJECT 
    await manager.connect(websocket)
    try:
        while True:
            mouse_obj = MOUSE_OBJECT
            ret = {
                'state' : int(mouse_obj.s),
                'x' : WINDOW_SIZE_WIDTH - int(mouse_obj.x),
                'y' : int(mouse_obj.y)
            }
            print(ret)
            await manager.broadcast(f"{json.dumps(ret)}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        # await manager.broadcast("A client disconnected")


    #while True:
    #    try:
    #        mouse_obj = MOUSE_OBJECT
    #        ret = {
    #            'state' : int(mouse_obj.s),
    #            'x' : WINDOW_SIZE_WIDTH - int(mouse_obj.x),
    #            'y' : int(mouse_obj.y)
    #        }
    #        print(ret)
    #        await manager.broadcast(f"{json.dumps(ret)}")
    #    except:
    #        print('Websocket error')
    #        manager.disconnect(websocket)