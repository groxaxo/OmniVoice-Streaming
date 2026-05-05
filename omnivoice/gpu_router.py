from __future__ import annotations

import argparse
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Iterable

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from omnivoice.gpu_pool import (
    PoolConfig,
    WorkerSpec,
    build_worker_specs,
    ensure_pool_dirs,
    prune_worker_processes,
    query_gpu_inventory,
    read_registry,
    worker_health,
)
from omnivoice.gpu_pool_celery import send_spawn_worker_task


LOG = logging.getLogger("omnivoice.gpu_router")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
READ_ONLY_PATHS = {
    "",
    "audio/models",
    "audio/voices",
    "credits",
    "models",
    "ui",
    "v1/audio/models",
    "v1/audio/voices",
    "v1/models",
}

# How much VRAM (MiB) to reserve per in-progress spawn when calculating
# whether a neighbouring slot on the same GPU can also be spawned.
_SPAWN_VRAM_RESERVE_MB = 10_000  # conservative: a worker uses ~8–9 GB at load time
_S1_KEEPALIVE_INTERVAL = 30.0  # seconds between keepalive polls


class OmniVoicePoolManager:
    def __init__(self, config: PoolConfig) -> None:
        self.config = config
        self.worker_specs = build_worker_specs(config)
        self._condition = asyncio.Condition()
        self._spawn_locks: dict[str, asyncio.Lock] = {
            spec.worker_key: asyncio.Lock() for spec in self.worker_specs
        }
        self._in_flight: dict[str, int] = {
            spec.worker_key: 0 for spec in self.worker_specs
        }
        # VRAM (MiB) currently reserved by in-progress spawns, keyed by GPU UUID.
        self._reserved_vram: dict[str, int] = {}
        self._client: httpx.AsyncClient | None = None
        self._keepalive_task: asyncio.Task | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def startup(self) -> None:
        if not self.worker_specs:
            raise RuntimeError(
                "No eligible OmniVoice GPU matched the current configuration"
            )
        ensure_pool_dirs(self.config)
        prune_worker_processes(self.config)
        # HTTP/2 + tuned connection pool dramatically reduces per-request
        # framing overhead when streaming long MP3 bodies through the router.
        try:
            self._client = httpx.AsyncClient(
                timeout=self.config.backend_timeout,
                http2=True,
                limits=httpx.Limits(
                    max_connections=128,
                    max_keepalive_connections=64,
                    keepalive_expiry=300.0,
                ),
            )
        except Exception:  # pragma: no cover — h2 not available
            self._client = httpx.AsyncClient(timeout=self.config.backend_timeout)

        # Pre-warm S1 slots (slot_index == 0) on matching GPUs.
        s1_specs = self._s1_specs()
        if s1_specs:
            LOG.info(
                "Pre-warming %d S1 slot(s): %s",
                len(s1_specs),
                [s.worker_key for s in s1_specs],
            )
            await asyncio.gather(
                *[self._ensure_worker_ready_bg(spec) for spec in s1_specs]
            )

        # Start S1 keepalive background loop.
        self._keepalive_task = asyncio.ensure_future(self._s1_keepalive_loop())

    async def shutdown(self) -> None:
        if self._keepalive_task is not None:
            self._keepalive_task.cancel()
            try:
                await self._keepalive_task
            except asyncio.CancelledError:
                pass
            self._keepalive_task = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    # ------------------------------------------------------------------
    # S1 keepalive
    # ------------------------------------------------------------------

    def _s1_specs(self) -> list[WorkerSpec]:
        """Return slot-0 specs whose GPU name matches the S1 filter."""
        filt = self.config.s1_gpu_name_filter.lower()
        if not filt:
            return []
        # We need the GPU name — query inventory once per call (cheap).
        try:
            inventory = query_gpu_inventory(list(self.config.gpu_uuids))
        except Exception:
            return []
        return [
            spec
            for spec in self.worker_specs
            if spec.slot_index == 0
            and filt
            in (
                inventory.get(spec.uuid, None) and inventory[spec.uuid].name or ""
            ).lower()
        ]

    async def _s1_keepalive_loop(self) -> None:
        """Periodically re-spawn S1 slots if they have gone idle."""
        while True:
            try:
                await asyncio.sleep(_S1_KEEPALIVE_INTERVAL)
                s1_specs = await asyncio.to_thread(self._s1_specs_sync)
                for spec in s1_specs:
                    healthy = await asyncio.to_thread(worker_health, spec.port, 1.0)
                    if not healthy:
                        LOG.info(
                            "S1 keepalive: slot %s (port %s) is down — re-spawning",
                            spec.worker_key,
                            spec.port,
                        )
                        asyncio.ensure_future(self._ensure_worker_ready_bg(spec))
            except asyncio.CancelledError:
                raise
            except Exception:
                LOG.exception("S1 keepalive loop error (will retry)")

    def _s1_specs_sync(self) -> list[WorkerSpec]:
        """Thread-safe version of _s1_specs (no asyncio.to_thread nesting)."""
        filt = self.config.s1_gpu_name_filter.lower()
        if not filt:
            return []
        try:
            inventory = query_gpu_inventory(list(self.config.gpu_uuids))
        except Exception:
            return []
        return [
            spec
            for spec in self.worker_specs
            if spec.slot_index == 0
            and filt
            in (
                inventory.get(spec.uuid, None) and inventory[spec.uuid].name or ""
            ).lower()
        ]

    # ------------------------------------------------------------------
    # Snapshot / health
    # ------------------------------------------------------------------

    async def snapshot(self) -> dict[str, object]:
        inventory = await asyncio.to_thread(
            query_gpu_inventory,
            list(self.config.gpu_uuids),
        )
        registry = await asyncio.to_thread(read_registry, self.config)
        workers = []
        for spec in self.worker_specs:
            health = await asyncio.to_thread(worker_health, spec.port, 0.5)
            gpu = inventory.get(spec.uuid)
            record = registry.get("workers", {}).get(spec.worker_key, {})
            workers.append(
                {
                    "gpu": {
                        "free_mb": gpu.free_mb if gpu else None,
                        "index": gpu.index if gpu else None,
                        "name": gpu.name if gpu else None,
                        "total_mb": gpu.total_mb if gpu else None,
                        "used_mb": gpu.used_mb if gpu else None,
                        "uuid": spec.uuid,
                    },
                    "in_flight": self._in_flight[spec.worker_key],
                    "ordinal": spec.ordinal,
                    "port": spec.port,
                    "ready": health is not None,
                    "registry": record,
                    "slot_index": spec.slot_index,
                    "worker_key": spec.worker_key,
                }
            )
        return {
            "acquire_timeout": self.config.acquire_timeout,
            "min_free_gpu_mb": self.config.min_free_gpu_mb,
            "router_host": self.config.router_host,
            "router_port": self.config.router_port,
            "s1_gpu_filter": self.config.s1_gpu_name_filter,
            "spawn_timeout": self.config.spawn_timeout,
            "status": "ok",
            "worker_idle_timeout": self.config.worker_idle_timeout,
            "workers": workers,
            "workers_per_gpu": self.config.workers_per_gpu,
        }

    # ------------------------------------------------------------------
    # Acquire / release
    # ------------------------------------------------------------------

    async def acquire_worker(self, require_slot: bool) -> WorkerSpec:
        deadline = time.monotonic() + self.config.acquire_timeout
        while True:
            inventory = await asyncio.to_thread(
                query_gpu_inventory,
                list(self.config.gpu_uuids),
            )
            ready = {
                spec.worker_key: bool(
                    await asyncio.to_thread(worker_health, spec.port, 0.5)
                )
                for spec in self.worker_specs
            }

            async with self._condition:
                free_ready = [
                    spec
                    for spec in self.worker_specs
                    if ready[spec.worker_key]
                    and (not require_slot or self._in_flight[spec.worker_key] == 0)
                ]
                if free_ready:
                    chosen = self._choose_best(free_ready, inventory)
                    if require_slot:
                        self._in_flight[chosen.worker_key] += 1
                    return chosen

                spawnable = self._find_spawnable(ready, inventory)
                if not spawnable and any(ready.values()) and require_slot:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise HTTPException(
                            status_code=503,
                            detail="All OmniVoice workers are busy and no GPU has enough free VRAM for a new slot",
                        )
                    try:
                        await asyncio.wait_for(
                            self._condition.wait(),
                            timeout=min(remaining, 1.0),
                        )
                    except TimeoutError:
                        pass
                    continue

            if spawnable:
                for spec in spawnable:
                    asyncio.ensure_future(self._ensure_worker_ready_bg(spec))
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise HTTPException(
                        status_code=503,
                        detail="No OmniVoice GPU is currently eligible for a worker",
                    )
                await asyncio.sleep(min(remaining, 1.0))
                continue

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HTTPException(
                    status_code=503,
                    detail="No OmniVoice GPU is currently eligible for a worker",
                )
            await asyncio.sleep(min(remaining, 1.0))

    async def release_worker(self, spec: WorkerSpec) -> None:
        async with self._condition:
            current = self._in_flight[spec.worker_key]
            self._in_flight[spec.worker_key] = max(0, current - 1)
            self._condition.notify_all()

    # ------------------------------------------------------------------
    # Spawning helpers
    # ------------------------------------------------------------------

    def _find_spawnable(
        self,
        ready: dict[str, bool],
        inventory: dict[str, object],
    ) -> list[WorkerSpec]:
        """
        Return specs that are not yet running and have enough free VRAM,
        accounting for VRAM already reserved by concurrent in-progress spawns
        on the same GPU.
        """
        result = []
        # Per-GPU effective free VRAM = reported free − already reserved.
        effective_free: dict[str, int] = {}
        for uuid in self.config.gpu_uuids:
            gpu = inventory.get(uuid)
            if gpu is None:
                effective_free[uuid] = 0
            else:
                effective_free[uuid] = gpu.free_mb - self._reserved_vram.get(uuid, 0)

        for spec in self.worker_specs:
            if ready[spec.worker_key]:
                continue
            if effective_free.get(spec.uuid, 0) >= self.config.min_free_gpu_mb:
                result.append(spec)
                # Reserve VRAM so that the next spec on the same GPU sees a
                # reduced effective free figure and doesn't over-subscribe.
                effective_free[spec.uuid] = (
                    effective_free[spec.uuid] - _SPAWN_VRAM_RESERVE_MB
                )
        return result

    def _choose_best(
        self,
        candidates: Iterable[WorkerSpec],
        inventory: dict[str, object],
    ) -> WorkerSpec:
        return sorted(
            candidates,
            key=lambda spec: (
                self._in_flight[spec.worker_key],
                -(inventory[spec.uuid].free_mb if spec.uuid in inventory else 0),
                spec.port,
            ),
        )[0]

    async def _ensure_worker_ready(self, spec: WorkerSpec) -> None:
        if await asyncio.to_thread(worker_health, spec.port, 0.5):
            return
        spawn_lock = self._spawn_locks[spec.worker_key]
        async with spawn_lock:
            if await asyncio.to_thread(worker_health, spec.port, 0.5):
                return
            LOG.info(
                "Requesting OmniVoice worker spawn: key=%s port=%s gpu=%s slot=%d",
                spec.worker_key,
                spec.port,
                spec.uuid,
                spec.slot_index,
            )
            # Reserve VRAM for the duration of the spawn so concurrent
            # _find_spawnable calls on the same GPU don't over-subscribe.
            self._reserved_vram[spec.uuid] = (
                self._reserved_vram.get(spec.uuid, 0) + _SPAWN_VRAM_RESERVE_MB
            )
            try:
                await asyncio.to_thread(send_spawn_worker_task, spec.worker_key)
                deadline = time.monotonic() + self.config.spawn_timeout
                while time.monotonic() < deadline:
                    if await asyncio.to_thread(worker_health, spec.port, 1.0):
                        return
                    await asyncio.sleep(1.0)
            finally:
                self._reserved_vram[spec.uuid] = max(
                    0,
                    self._reserved_vram.get(spec.uuid, 0) - _SPAWN_VRAM_RESERVE_MB,
                )
        raise HTTPException(
            status_code=503,
            detail=(
                f"Timed out while starting OmniVoice worker {spec.worker_key} "
                f"on port {spec.port}"
            ),
        )

    async def _ensure_worker_ready_bg(self, spec: WorkerSpec) -> None:
        """Fire-and-forget parallel spawn; logs failures instead of raising."""
        try:
            await self._ensure_worker_ready(spec)
        except Exception:
            LOG.exception(
                "Background worker spawn failed for GPU %s port %s",
                spec.uuid,
                spec.port,
            )

    # ------------------------------------------------------------------
    # Proxy
    # ------------------------------------------------------------------

    async def proxy(self, request: Request, full_path: str) -> Response:
        if self._client is None:
            raise RuntimeError("HTTP client not initialized")

        normalized_path = full_path.strip("/")
        require_slot = not (
            request.method.upper() == "GET" and normalized_path in READ_ONLY_PATHS
        )
        worker = await self.acquire_worker(require_slot=require_slot)
        body = await request.body()
        target_url = f"http://127.0.0.1:{worker.port}/{normalized_path}"
        if not normalized_path:
            target_url = f"http://127.0.0.1:{worker.port}/"

        headers = {
            key: value
            for key, value in request.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
        }

        # Stream the upstream response so big audio payloads (multi-MB MP3s)
        # don't have to be fully buffered in router RAM before the client sees
        # the first byte. The upstream slot is held until the stream finishes.
        try:
            req = self._client.build_request(
                method=request.method,
                url=target_url,
                content=body,
                headers=headers,
                params=request.query_params,
            )
            upstream = await self._client.send(req, stream=True, follow_redirects=False)
        except httpx.HTTPError as exc:
            if require_slot:
                await self.release_worker(worker)
            raise HTTPException(
                status_code=503,
                detail=f"OmniVoice backend request failed: {exc}",
            ) from exc

        response_headers = {
            key: value
            for key, value in upstream.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "content-type"
        }
        response_headers["X-OmniVoice-Backend-Port"] = str(worker.port)
        response_headers["X-OmniVoice-Backend-GPU-UUID"] = worker.uuid
        response_headers["X-OmniVoice-Backend-Slot"] = str(worker.slot_index)

        async def _body_iter():
            try:
                async for chunk in upstream.aiter_raw():
                    yield chunk
            finally:
                try:
                    await upstream.aclose()
                finally:
                    if require_slot:
                        await self.release_worker(worker)

        return StreamingResponse(
            _body_iter(),
            headers=response_headers,
            media_type=upstream.headers.get("content-type"),
            status_code=upstream.status_code,
        )


CONFIG = PoolConfig.from_env()
POOL = OmniVoicePoolManager(CONFIG)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=False,
    )
    LOG.info(
        "Starting OmniVoice router on %s:%s — GPUs=%s workers_per_gpu=%d s1_filter=%r",
        CONFIG.router_host,
        CONFIG.router_port,
        ",".join(CONFIG.gpu_uuids),
        CONFIG.workers_per_gpu,
        CONFIG.s1_gpu_name_filter,
    )
    await POOL.startup()
    yield
    await POOL.shutdown()


app = FastAPI(title="OmniVoice GPU Router", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
@app.get("/v1/health")
async def health() -> dict[str, object]:
    return await POOL.snapshot()


@app.get("/pool/status")
async def pool_status() -> dict[str, object]:
    return await POOL.snapshot()


@app.api_route(
    "/{full_path:path}",
    methods=["DELETE", "GET", "HEAD", "OPTIONS", "PATCH", "POST", "PUT"],
)
async def proxy_request(full_path: str, request: Request) -> Response:
    return await POOL.proxy(request, full_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniVoice GPU-aware router")
    parser.add_argument("--host", default=CONFIG.router_host)
    parser.add_argument("--port", type=int, default=CONFIG.router_port)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    import os
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        access_log=os.getenv("OMNIVOICE_ROUTER_ACCESS_LOG", "1").lower() not in {"0", "false", "no"},
        workers=1,
        loop=os.getenv("OMNIVOICE_UVICORN_LOOP", "uvloop"),
        http=os.getenv("OMNIVOICE_UVICORN_HTTP", "httptools"),
    )


if __name__ == "__main__":
    main()
