import logging

# from libmambapy import LogLevel
# from sympy import hyper
# from masha.config import app_config
from fastapi import FastAPI, Request
from contextlib import asynccontextmanager


from corelog import register, Handlers
import os

register(os.environ.get("MASHA_LOG_LEVEL", "INFO"), handler_type=Handlers.RICH)

from masha.image.huggingface.stablediffusion import StableDiffusion


@asynccontextmanager
async def lifespan(app: FastAPI):
    from masha.firebase.db import OptionsDb
    from masha.image import OPTIONS as ImageOptions
    from masha.code.instruct import InstructOption
    OptionsDb().options(
        code={"instruct": InstructOption.values()},
        image=ImageOptions
    )    
    yield
    logging.info("lifespan")


def create_app():
    app = FastAPI(
        title="masha@cacko.net",
        lifespan=lifespan
    )

    @app.middleware("http")
    async def check_superuser(request: Request, call_next):
        try:
            assert request.headers.get("X-SuperUser")
            StableDiffusion.is_superuser = True
        except AssertionError:
            StableDiffusion.is_superuser = False
        response = await call_next(request)
        return response

    from masha.code import router as code_router
    from masha.chat import router as chat_router
    from masha.image import router as image_router
    from masha.knowledge_base import router as kb_router
    from masha.name import router as name_router
    from masha.text import router as text_router
    from masha.translate import router as translate_router
    from masha.hue import router as hue_router
    from masha.geo import router as geo_router

    app.include_router(code_router, prefix="/code", tags=["code"])
    app.include_router(chat_router, prefix="/chat", tags=["chat"])
    app.include_router(image_router, prefix="/image", tags=["image"])
    app.include_router(kb_router, prefix="/kb", tags=["kb"])
    app.include_router(name_router, prefix="/name", tags=["name"])
    app.include_router(text_router, prefix="/text", tags=["text"])
    app.include_router(translate_router, prefix="/translate", tags=["translate"])
    app.include_router(hue_router, prefix="/hue", tags=["hue"])
    app.include_router(geo_router, prefix="/geo", tags=["geo"])
    return app


# def serve():
#     uvloop.install()
#     server_config = Config.from_mapping(
#         bind=f"{app_config.server.host}:{app_config.server.port}",
#         accesslog="-",
#         errorlog="-",
#         LogLevel=environ.get("MASHA_LOG_LEVEL", "INFO"),
#         workers=app_config.server.workers,
#         worker_class="uvloop"
#     )
#     asyncio.run(hyper_serve(create_app(), server_config))
#     # server_config = Config(
#     #     app=create_app(),
#     #     use_colors=True,
#     #     log_level=logging._nameToLevel.get(
#     #         environ.get("MASHA_LOG_LEVEL", "INFO"),
#     #     ),
#     #     **app_config.server.model_dump()
#     # )
#     # server = Server(server_config)
#     # server.run()


# # def serve():
# #     server_config = Config.from_mapping(
# #         bind=f"{app_config.server.host}:{app_config.server.port}",
# #         worker_class="trio",
# #         accesslog="-",
# #         errorlog="-",
# #         loglevel=environ.get("MASHA_LOG_LEVEL", "INFO"),
# #     )
# #     asyncio.run(hypercorn_serve(create_app(), server_config))
