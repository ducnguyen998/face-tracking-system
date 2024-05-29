from fastapi import (
    APIRouter
)

from routes.route_mouse import (
    router as _route_mouse
)

router = APIRouter()
router.include_router(_route_mouse, prefix="/mouse")