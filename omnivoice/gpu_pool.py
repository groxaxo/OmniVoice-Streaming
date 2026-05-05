from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_runtime import (
    GPUStat,
    choose_single_gpu,
    locked_json_registry,
    parse_uuid_list,
    process_exists,
    prune_registry_workers,
    query_gpu_inventory,
    read_json_registry,
    spawn_logged_process,
)


LOG = logging.getLogger("omnivoice.gpu_pool")


@dataclass(frozen=True, slots=True)
class WorkerSpec:
    ordinal: int  # global ordinal across all GPU slots (determines port)
    slot_index: int  # slot index within this GPU (0 = primary / s1 slot)
    uuid: str  # GPU UUID — used for CUDA_VISIBLE_DEVICES
    port: int

    @property
    def label(self) -> str:
        return f"worker-{self.ordinal + 1}"

    @property
    def worker_key(self) -> str:
        """Unique per-worker identifier. Port is always unique across the pool."""
        return str(self.port)


@dataclass(frozen=True, slots=True)
class PoolConfig:
    router_host: str
    router_port: int
    gpu_uuids: tuple[str, ...]
    workers_per_gpu: int
    worker_base_port: int
    min_free_gpu_mb: int
    s1_gpu_name_filter: str  # GPU name substring → keep slot-0 warm (e.g. "3060")
    acquire_timeout: float
    spawn_timeout: float
    worker_idle_timeout: float
    backend_timeout: float
    state_dir: Path
    registry_path: Path
    broker_dir: Path
    log_dir: Path
    python_bin: str
    repo_root: Path
    model_id: str | None
    gpu_name_filter: str | None

    @classmethod
    def from_env(cls) -> PoolConfig:
        repo_root = Path(
            os.getenv("OMNIVOICE_REPO_ROOT", "/home/op/OmniVoice")
        ).expanduser()
        state_dir = Path(
            os.getenv("OMNIVOICE_STATE_DIR", "~/.cache/omnivoice-pool")
        ).expanduser()
        registry_path = Path(
            os.getenv(
                "OMNIVOICE_REGISTRY_PATH",
                str(state_dir / "workers.json"),
            )
        ).expanduser()
        broker_dir = Path(
            os.getenv(
                "OMNIVOICE_CELERY_BROKER_DIR",
                str(state_dir / "celery-broker"),
            )
        ).expanduser()
        log_dir = Path(
            os.getenv(
                "OMNIVOICE_WORKER_LOG_DIR",
                str(state_dir / "logs"),
            )
        ).expanduser()
        python_bin = os.getenv(
            "OMNIVOICE_PYTHON_BIN",
            sys.executable,
        )
        model_id = os.getenv("OMNIVOICE_MODEL_ID") or None
        gpu_name_filter = os.getenv("OMNIVOICE_GPU_NAME_FILTER") or None
        preferred_gpu_uuid = os.getenv("OMNIVOICE_GPU_UUID", "").strip()
        configured_gpu_uuids = tuple(
            parse_uuid_list(os.getenv("OMNIVOICE_GPU_UUIDS", ""))
        )

        requested_gpu_uuids = (
            (preferred_gpu_uuid,)
            if preferred_gpu_uuid
            else (configured_gpu_uuids if configured_gpu_uuids else None)
        )
        inventory = query_gpu_inventory(
            target_uuids=requested_gpu_uuids,
            name_filter=gpu_name_filter,
        )

        if configured_gpu_uuids:
            active_uuids = tuple(
                uuid for uuid in configured_gpu_uuids if uuid in inventory
            )
        elif preferred_gpu_uuid:
            active_uuids = (preferred_gpu_uuid,)
        else:
            selected_gpu = choose_single_gpu(
                inventory,
                preferred_uuid=None,
                allowed_uuids=None,
            )
            active_uuids = (selected_gpu.uuid,)

        return cls(
            router_host=os.getenv("OMNIVOICE_ROUTER_HOST", "0.0.0.0"),
            router_port=int(os.getenv("OMNIVOICE_ROUTER_PORT", "6655")),
            gpu_uuids=active_uuids,
            workers_per_gpu=int(os.getenv("OMNIVOICE_WORKERS_PER_GPU", "1")),
            worker_base_port=int(os.getenv("OMNIVOICE_WORKER_BASE_PORT", "6656")),
            min_free_gpu_mb=int(os.getenv("OMNIVOICE_MIN_FREE_GPU_MB", "8000")),
            s1_gpu_name_filter=os.getenv("OMNIVOICE_S1_GPU_FILTER", "3060"),
            acquire_timeout=float(os.getenv("OMNIVOICE_ACQUIRE_TIMEOUT", "180")),
            spawn_timeout=float(os.getenv("OMNIVOICE_SPAWN_TIMEOUT", "180")),
            worker_idle_timeout=float(
                os.getenv("OMNIVOICE_WORKER_IDLE_TIMEOUT", "300")
            ),
            backend_timeout=float(os.getenv("OMNIVOICE_BACKEND_TIMEOUT", "600")),
            state_dir=state_dir,
            registry_path=registry_path,
            broker_dir=broker_dir,
            log_dir=log_dir,
            python_bin=python_bin,
            repo_root=repo_root,
            model_id=model_id,
            gpu_name_filter=gpu_name_filter,
        )

    @property
    def broker_queue_dir(self) -> Path:
        return self.broker_dir / "queue"

    @property
    def broker_in_dir(self) -> Path:
        return self.broker_queue_dir

    @property
    def broker_out_dir(self) -> Path:
        return self.broker_queue_dir

    @property
    def broker_processed_dir(self) -> Path:
        return self.broker_dir / "processed"

    @property
    def broker_control_dir(self) -> Path:
        return self.broker_dir / "control"

    def broker_transport_options(self) -> dict[str, str]:
        return {
            "data_folder_in": str(self.broker_in_dir),
            "data_folder_out": str(self.broker_out_dir),
            "data_folder_processed": str(self.broker_processed_dir),
            "control_folder": str(self.broker_control_dir),
        }


def ensure_pool_dirs(config: PoolConfig) -> None:
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.log_dir.mkdir(parents=True, exist_ok=True)
    config.broker_queue_dir.mkdir(parents=True, exist_ok=True)
    config.broker_processed_dir.mkdir(parents=True, exist_ok=True)
    config.broker_control_dir.mkdir(parents=True, exist_ok=True)
    config.registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not config.registry_path.exists():
        config.registry_path.write_text('{"workers": {}}\n', encoding="utf-8")


def build_worker_specs(config: PoolConfig) -> list[WorkerSpec]:
    """
    Build the full list of worker slot specs across all GPUs.

    Layout: ordinal increments globally; slot_index resets per GPU.
    Port = worker_base_port + ordinal (unique across the entire pool).

    Example: 5 GPUs, workers_per_gpu=3, base_port=6656
      GPU-0: ordinals 0,1,2  → ports 6656,6657,6658  slot_index 0,1,2
      GPU-1: ordinals 3,4,5  → ports 6659,6660,6661  slot_index 0,1,2
      GPU-2: ordinals 6,7,8  → ports 6662,6663,6664  slot_index 0,1,2
      ...
    """
    specs = []
    ordinal = 0
    for uuid in config.gpu_uuids:
        for slot_idx in range(config.workers_per_gpu):
            port = config.worker_base_port + ordinal
            specs.append(
                WorkerSpec(
                    ordinal=ordinal,
                    slot_index=slot_idx,
                    uuid=uuid,
                    port=port,
                )
            )
            ordinal += 1
    return specs


def prune_worker_processes(config: PoolConfig) -> None:
    ensure_pool_dirs(config)
    desired_keys = {spec.worker_key for spec in build_worker_specs(config)}
    prune_registry_workers(
        config.registry_path,
        desired_keys=desired_keys,
        initial_data={"workers": {}},
    )


def worker_health(port: int, timeout: float = 1.0) -> dict[str, Any] | None:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/health",
        headers={"Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                return None
            payload = response.read().decode("utf-8")
    except (OSError, TimeoutError, urllib.error.URLError):
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if data.get("status") != "ok":
        return None
    return data


def locked_registry(config: PoolConfig) -> Any:
    ensure_pool_dirs(config)
    return locked_json_registry(config.registry_path, initial_data={"workers": {}})


def read_registry(config: PoolConfig) -> dict[str, Any]:
    ensure_pool_dirs(config)
    return read_json_registry(config.registry_path, initial_data={"workers": {}})


def build_worker_command(config: PoolConfig, spec: WorkerSpec) -> list[str]:
    command = [
        config.python_bin,
        "-m",
        "omnivoice.openai_tts_server",
        "--host",
        "0.0.0.0",
        "--port",
        str(spec.port),
        "--device",
        "cuda:0",
        "--idle-timeout",
        str(config.worker_idle_timeout),
    ]
    if config.model_id:
        command.extend(["--model-id", config.model_id])
    return command


def build_worker_env(config: PoolConfig, spec: WorkerSpec) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = spec.uuid
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    env.setdefault("NCCL_IB_DISABLE", "1")
    env.setdefault("NCCL_P2P_DISABLE", "0")
    env["OMNIVOICE_HOST"] = "0.0.0.0"
    env["OMNIVOICE_PORT"] = str(spec.port)
    env["OMNIVOICE_IDLE_TIMEOUT"] = str(config.worker_idle_timeout)
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    env["OMNIVOICE_POOL_WORKER_UUID"] = spec.uuid
    env["OMNIVOICE_POOL_WORKER_PORT"] = str(spec.port)
    return env


def spawn_worker_process(spec: WorkerSpec, config: PoolConfig) -> dict[str, Any]:
    ensure_pool_dirs(config)
    if worker_health(spec.port, timeout=1.0):
        with locked_registry(config) as data:
            record = data.setdefault("workers", {}).get(spec.worker_key, {})
            record.update(
                {
                    "port": spec.port,
                    "uuid": spec.uuid,
                    "worker_key": spec.worker_key,
                    "slot_index": spec.slot_index,
                    "status": "healthy",
                    "last_seen_at": time.time(),
                }
            )
            data["workers"][spec.worker_key] = record
            return dict(record)

    with locked_registry(config) as data:
        record = data.setdefault("workers", {}).get(spec.worker_key, {})
        if process_exists(record.get("pid")) and worker_health(spec.port, timeout=1.0):
            record["status"] = "healthy"
            record["last_seen_at"] = time.time()
            data["workers"][spec.worker_key] = record
            return dict(record)

        log_path = config.log_dir / f"worker-{spec.port}.log"
        process = spawn_logged_process(
            build_worker_command(config, spec),
            cwd=config.repo_root,
            env=build_worker_env(config, spec),
            log_path=log_path,
        )

        record = {
            "command": build_worker_command(config, spec),
            "gpu_uuid": spec.uuid,
            "worker_key": spec.worker_key,
            "slot_index": spec.slot_index,
            "label": spec.label,
            "last_seen_at": time.time(),
            "log_path": str(log_path),
            "pid": process.pid,
            "port": spec.port,
            "spawned_at": time.time(),
            "status": "starting",
            "uuid": spec.uuid,
        }
        data["workers"][spec.worker_key] = record
        return dict(record)
