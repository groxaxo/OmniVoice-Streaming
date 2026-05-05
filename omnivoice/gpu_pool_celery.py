from __future__ import annotations

import logging
import os
import socket

from celery import Celery

from omnivoice.gpu_pool import (
    PoolConfig,
    build_worker_specs,
    ensure_pool_dirs,
    prune_worker_processes,
    query_gpu_inventory,
    spawn_worker_process,
)


LOG = logging.getLogger("omnivoice.gpu_pool_celery")
CONFIG = PoolConfig.from_env()
ensure_pool_dirs(CONFIG)
prune_worker_processes(CONFIG)
WORKER_SPEC_MAP = {spec.worker_key: spec for spec in build_worker_specs(CONFIG)}

celery_app = Celery("omnivoice-gpu-pool", broker="filesystem://")
celery_app.conf.update(
    accept_content=["json"],
    broker_transport_options=CONFIG.broker_transport_options(),
    task_default_queue="omnivoice-spawn",
    task_ignore_result=True,
    task_serializer="json",
    worker_prefetch_multiplier=1,
)


@celery_app.task(name="omnivoice.spawn_worker")
def spawn_worker(worker_key: str) -> dict[str, object]:
    spec = WORKER_SPEC_MAP.get(worker_key)
    if spec is None:
        raise ValueError(f"Unknown OmniVoice worker key: {worker_key}")

    inventory = query_gpu_inventory([spec.uuid])
    gpu = inventory.get(spec.uuid)
    if gpu is None:
        raise RuntimeError(f"Configured OmniVoice GPU {spec.uuid} is not visible")
    if gpu.free_mb < CONFIG.min_free_gpu_mb:
        raise RuntimeError(
            f"GPU {spec.uuid} has only {gpu.free_mb} MiB free; "
            f"{CONFIG.min_free_gpu_mb} MiB required to spawn OmniVoice"
        )

    record = spawn_worker_process(spec, CONFIG)
    LOG.info(
        "Spawned OmniVoice worker %s on port %s (gpu=%s, pid=%s)",
        worker_key,
        record.get("port"),
        spec.uuid,
        record.get("pid"),
    )
    return {
        "pid": record.get("pid"),
        "port": record.get("port"),
        "status": record.get("status"),
        "worker_key": worker_key,
        "gpu_uuid": spec.uuid,
    }


def send_spawn_worker_task(worker_key: str) -> None:
    ensure_pool_dirs(CONFIG)
    celery_app.send_task(
        "omnivoice.spawn_worker",
        kwargs={"worker_key": worker_key},
        queue="omnivoice-spawn",
    )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    hostname = socket.gethostname()
    loglevel = os.getenv("OMNIVOICE_CELERY_LOGLEVEL", "INFO")
    # Use threads pool so multiple spawn tasks can execute concurrently —
    # one thread per configured worker so all can be started in parallel.
    total_workers = len(build_worker_specs(CONFIG))
    concurrency = str(max(total_workers, 4))
    celery_app.worker_main(
        [
            "worker",
            "--loglevel",
            loglevel,
            "--concurrency",
            concurrency,
            "--pool",
            "threads",
            "--hostname",
            f"omnivoice-spawner@{hostname}",
            "--queues",
            "omnivoice-spawn",
            "--without-gossip",
            "--without-mingle",
            "--without-heartbeat",
        ]
    )


if __name__ == "__main__":
    main()
