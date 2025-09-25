import logging
from requests_toolbelt import MultipartEncoder
from pathlib import Path
from fastapi import Response, UploadFile
from corefile import TempPath, find_mime_type
from typing import Optional


def make_response(image_path: Optional[Path] = None, message: Optional[str] = None):
    if image_path:
        return make_multipart_response(image_path, message)
    return {"response": message}


def make_multipart_response(image_path: Path, message: Optional[str] = None):
    assert image_path.exists()
    mime = find_mime_type(image_path.as_posix())
    assert mime
    m = MultipartEncoder(
        fields={
            "message": message if message else "",
            "file": (image_path.name, image_path.open("rb"), mime),
        }
    )
    logging.info(m.content_type)
    return Response(m.to_string(), media_type=m.content_type)


async def uploaded_file(file: UploadFile) -> TempPath:
    tmp_path = TempPath(f"uploaded_file_{file.filename}")
    logging.warning(tmp_path)
    data = await file.read()
    tmp_path.write_bytes(data)
    return tmp_path
