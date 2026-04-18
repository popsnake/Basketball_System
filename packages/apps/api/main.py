from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from apps.api.routers import chat, demo, kb, scoring
from models_dl.config import load_model_yaml, repo_root

app = FastAPI(title="ShootCoach API", version="0.1.0")
app.include_router(scoring.router)
app.include_router(chat.router)
app.include_router(demo.router)
app.include_router(kb.router)

static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/version")
def version() -> dict[str, str]:
    cfg = load_model_yaml()
    p = repo_root() / cfg["paths"]["weights_h5"]
    return {
        "config": "configs/model.yaml",
        "weights_present": str(p.is_file()),
        "weights_path": str(p),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


def main() -> None:
    import uvicorn

    uvicorn.run("apps.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
