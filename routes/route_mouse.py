import sys
import json

from pathlib import (
    Path
)


from fastapi import (
    APIRouter,
    Response
)

sys.path.append(str(Path(__file__).parent))

router = APIRouter()

@router.post("/report")
async def mouse_report():
    ret = {
        'state' : 1,
        'x' : 115,
        'y' : 215
    }
    return Response(content=json.dumps(ret), media_type='content/json')