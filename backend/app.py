"""Loopback-only Cadence demo API. No inference starts at module import."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import asyncio
import logging
import math
import threading
import time
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.config import MODEL_ID, REVISION, OPTION_LIMITS, PROFILES
from backend.validation import compact_json_length, utf16_length

MAX_BODY = 65536
log = logging.getLogger("cadence.demo")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    type: Literal["choice", "noul", "score"]
    instructions: str = Field(min_length=1, max_length=2000)
    criteria: dict[str, str | None] | list[str]

    @model_validator(mode="after")
    def criteria_match_type(self):
        if self.type == "score":
            if not isinstance(self.criteria, list) or not 2 <= len(self.criteria) <= 32:
                raise ValueError("Score needs 2–32 ordered levels.")
            if any(
                not text.strip() or utf16_length(text) > 500 for text in self.criteria
            ):
                raise ValueError(
                    "Each score level needs a description of up to 500 characters."
                )
        else:
            if not isinstance(self.criteria, dict):
                raise ValueError("Choice and Boolean criteria must be objects.")
            if self.type == "noul" and set(self.criteria) != {"false", "true"}:
                raise ValueError("Boolean criteria need exactly false and true keys.")
            if self.type == "choice" and not 2 <= len(self.criteria) <= 32:
                raise ValueError("Choice needs 2–32 options.")
            for key, value in self.criteria.items():
                if not key.strip() or utf16_length(key) > 64:
                    raise ValueError(
                        "Option IDs must be nonempty and at most 64 characters."
                    )
                if value is not None and utf16_length(value) > 500:
                    raise ValueError(
                        "Option descriptions are limited to 500 characters."
                    )
        if utf16_length(self.instructions) > 2000:
            raise ValueError("Instructions are limited to 2,000 UTF-16 units.")
        if not self.instructions.strip():
            raise ValueError("Instructions must not be blank.")
        return self


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    state: str | dict | list
    questions: dict[str, Question]

    @model_validator(mode="after")
    def bounded_request(self):
        if set(self.questions) != {"decision"}:
            raise ValueError("Provide exactly one question named decision.")
        if isinstance(self.state, str):
            if not self.state.strip():
                raise ValueError("State must not be blank.")
            size = utf16_length(self.state)
        else:
            if not self.state:
                raise ValueError("JSON state must not be empty.")
            try:
                size = compact_json_length(self.state)
            except (ValueError, RecursionError):
                raise ValueError("JSON state must be finite and valid.")
        if size > 12000:
            raise ValueError(
                "State is limited to 12,000 UTF-16 units (compact JSON for objects/arrays)."
            )
        return self


class BodyLimitMiddleware:
    """Bound bytes before JSON parsing, including chunked requests."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST":
            return await self.app(scope, receive, send)
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > MAX_BODY:
                response = JSONResponse(
                    {"detail": "Request exceeds the 64 KiB demo limit."},
                    status_code=413,
                )
                return await response(scope, receive, send)
            if not message.get("more_body", False):
                break
        delivered = False

        async def bounded_receive():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return await receive()

        await self.app(scope, bounded_receive, send)


class ModelService:
    def __init__(self, loader, device):
        self.loader = loader
        self.device = device
        self.profile = PROFILES[device]
        self.model = None
        self.status = "cold"
        self.lock = threading.Lock()
        self.calls = deque()
        # A dedicated worker owns model construction and subsequent CUDA calls.
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="cadence")

    def run(self, request):
        try:
            if self.model is None:
                self.status = "loading"
                try:
                    self.model = self.loader()
                except Exception:
                    self.status = "error"
                    log.exception("Model load failed")
                    raise HTTPException(
                        503,
                        f"Model could not load for {self.device}. Check local device support, pinned dependencies and cache setup in README, then retry.",
                    )
            self.status = "ready"
            try:
                result = self.model.predict(request)
            except ValueError:
                raise HTTPException(
                    422,
                    "Model rejected this input. Shorten the state, instructions or option descriptions and retry.",
                )
            except Exception:
                log.exception("Inference failed")
                raise HTTPException(
                    503,
                    f"Inference failed on {self.device}. Check the backend terminal and setup guide, then retry.",
                )
            if not isinstance(result, dict):
                raise HTTPException(
                    502, "Model returned an invalid result. No result was displayed."
                )
            keys = result.get("native_option_ids", [])
            probabilities = result.get("probabilities", {})
            question = request["questions"]["decision"]
            expected = (
                list(question["criteria"])
                if question["type"] == "choice"
                else ["false", "true"]
                if question["type"] == "noul"
                else [str(i) for i in range(len(question["criteria"]))]
            )
            if (
                result.get("kind") != question["type"]
                or keys != expected
                or not isinstance(probabilities, dict)
                or set(probabilities) != set(expected)
                or any(
                    not isinstance(p, (int, float))
                    or not math.isfinite(p)
                    or not 0 <= p <= 1
                    for p in probabilities.values()
                )
                or not math.isclose(sum(probabilities.values()), 1, abs_tol=1e-5)
                or result.get("truncated") is not False
                or not isinstance(result.get("tokens"), int)
                or isinstance(result.get("tokens"), bool)
                or result["tokens"] < 1
                or not isinstance(result.get("seconds"), (int, float))
                or not math.isfinite(result["seconds"])
                or result["seconds"] < 0
            ):
                raise HTTPException(
                    502,
                    "Model returned an invalid distribution. No result was displayed.",
                )
            value = None
            if question["type"] == "noul":
                value = probabilities["true"]
            if question["type"] == "score":
                value = sum(float(k) * probabilities[k] for k in expected)
            return {
                "kind": question["type"],
                "native_option_ids": keys,
                "probabilities": probabilities,
                "value": value,
                "tokens": result["tokens"],
                "seconds": result["seconds"],
                "truncated": False,
                "model": MODEL_ID,
                "revision": REVISION,
                "device": self.device,
                "profile": self.profile,
            }
        finally:
            self.lock.release()


def create_app(loader=None, *, device=None):
    from backend.runtime import load_model, selected_device

    device = selected_device() if device is None else device
    if device not in PROFILES:
        raise ValueError("Device must be cpu or cuda.")
    loader = (lambda: load_model(device=device)) if loader is None else loader
    service = ModelService(loader, device)

    @asynccontextmanager
    async def lifespan(api):
        try:
            yield
        finally:
            await asyncio.to_thread(service.executor.shutdown, wait=True)
            service.model = None
            service.status = "cold"

    api = FastAPI(
        title="Cadence local demo",
        docs_url="/api/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    api.state.service = service
    api.add_middleware(BodyLimitMiddleware)

    @api.exception_handler(RequestValidationError)
    async def invalid_request(request, exc):
        # Pydantic's default response echoes input. Keep request text out of errors.
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(
            {
                "detail": "Invalid request. Check the state, instructions and criteria.",
                "errors": errors,
            },
            status_code=422,
        )

    @api.get("/api/health")
    def health():
        return {
            "status": service.status,
            "busy": service.lock.locked(),
            "model": MODEL_ID,
            "revision": REVISION,
            "device": service.device,
            "profile": service.profile,
            "inference": "real",
            "limits": {
                "state_characters": 12000,
                "options": OPTION_LIMITS[device],
                "body_bytes": MAX_BODY,
            },
        }

    @api.post("/api/predict")
    async def predict(request: DecisionRequest):
        if len(request.questions["decision"].criteria) > OPTION_LIMITS[device]:
            raise HTTPException(
                422,
                f"This {device} demo profile permits at most {OPTION_LIMITS[device]} options or levels.",
            )
        if not service.lock.acquire(blocking=False):
            raise HTTPException(
                429, "Backend is busy with another decision. Wait and retry."
            )
        now = time.monotonic()
        while service.calls and now - service.calls[0] >= 60:
            service.calls.popleft()
        if len(service.calls) >= 12:
            service.lock.release()
            raise HTTPException(
                429, "Demo limit: 12 decisions per minute. Wait a minute and retry."
            )
        service.calls.append(now)
        # Cancellation does not free the lock while the worker is still running.
        future = service.executor.submit(service.run, request.model_dump())
        return await asyncio.shield(asyncio.wrap_future(future))

    return api


app = create_app()
