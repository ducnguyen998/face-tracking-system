import sys 

from pathlib import (
    Path
)

from fastapi import (
    FastAPI
)

from middleware import (
    LogMiddleware, 
    setup_cors
)

from routes.base import (
    router
)

sys.path.append(str(Path(__file__).parent))

app = FastAPI()

app.add_middleware(LogMiddleware)
setup_cors(app)
app.include_router(router)
