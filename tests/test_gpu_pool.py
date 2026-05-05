import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from omnivoice.gpu_pool import GPUStat, PoolConfig, build_worker_specs, spawn_worker_process
from omnivoice.gpu_router import OmniVoicePoolManager


def _test_config(state_dir: Path) -> PoolConfig:
    return PoolConfig(
        router_host="127.0.0.1",
        router_port=6655,
        gpu_uuids=(
            "GPU-72f3cda5-5f26-bed3-6093-0608adb365e3",
            "GPU-cbfc8a5f-0df1-ca71-f704-0d09a707d2ac",
        ),
        worker_base_port=6656,
        min_free_gpu_mb=8000,
        acquire_timeout=0.25,
        spawn_timeout=0.25,
        worker_idle_timeout=300.0,
        backend_timeout=5.0,
        state_dir=state_dir,
        registry_path=state_dir / "workers.json",
        broker_dir=state_dir / "broker",
        log_dir=state_dir / "logs",
        python_bin="/usr/bin/python3",
        repo_root=Path("/home/op/OmniVoice"),
        model_id=None,
    )


class PoolHelperTests(unittest.TestCase):
    def test_build_worker_specs_assigns_sequential_ports(self) -> None:
        config = _test_config(Path("/tmp/omnivoice-pool-test"))

        specs = build_worker_specs(config)

        self.assertEqual([spec.port for spec in specs], [6656, 6657])
        self.assertEqual(specs[0].uuid, config.gpu_uuids[0])
        self.assertEqual(specs[1].uuid, config.gpu_uuids[1])

    def test_spawn_worker_process_uses_uuid_pinned_cuda_visibility(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            spec = build_worker_specs(config)[0]

            class _FakeProcess:
                pid = 424242

            with (
                patch("omnivoice.gpu_pool.worker_health", return_value=None),
                patch("omnivoice.gpu_pool.subprocess.Popen", return_value=_FakeProcess()) as mock_popen,
            ):
                record = spawn_worker_process(spec, config)

            kwargs = mock_popen.call_args.kwargs
            self.assertEqual(kwargs["cwd"], config.repo_root)
            self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], spec.uuid)
            self.assertEqual(kwargs["env"]["OMNIVOICE_PORT"], str(spec.port))
            self.assertIn("--device", kwargs["args"] if "args" in kwargs else mock_popen.call_args.args[0])
            self.assertEqual(record["pid"], 424242)

            registry = json.loads(config.registry_path.read_text(encoding="utf-8"))
            self.assertEqual(
                registry["workers"][spec.uuid]["gpu_uuid"],
                spec.uuid,
            )


class RouterSelectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_acquire_worker_prefers_idle_ready_worker_with_more_free_vram(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            manager = OmniVoicePoolManager(config)
            await manager.startup()
            try:
                inventory = {
                    config.gpu_uuids[0]: GPUStat(2, config.gpu_uuids[0], "RTX 3060", 12288, 2048, 9500),
                    config.gpu_uuids[1]: GPUStat(3, config.gpu_uuids[1], "RTX 3060", 12288, 3072, 8700),
                }
                with (
                    patch("omnivoice.gpu_router.query_gpu_inventory", return_value=inventory),
                    patch("omnivoice.gpu_router.worker_health", return_value={"status": "ok"}),
                ):
                    worker = await manager.acquire_worker(require_slot=True)
                self.assertEqual(worker.uuid, config.gpu_uuids[0])
                await manager.release_worker(worker)
            finally:
                await manager.shutdown()

    async def test_acquire_worker_spawns_second_worker_for_concurrent_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            manager = OmniVoicePoolManager(config)
            await manager.startup()
            try:
                first_uuid, second_uuid = config.gpu_uuids
                first_port = config.worker_base_port
                second_port = config.worker_base_port + 1
                inventory = {
                    first_uuid: GPUStat(2, first_uuid, "RTX 3060", 12288, 2500, 9100),
                    second_uuid: GPUStat(3, second_uuid, "RTX 3060", 12288, 1024, 10200),
                }
                spawned = {"requested": False}

                def _health(port: int, timeout: float = 0.5):
                    if port == first_port:
                        return {"status": "ok"}
                    if port == second_port and spawned["requested"]:
                        return {"status": "ok"}
                    return None

                def _spawn(_worker_uuid: str) -> None:
                    spawned["requested"] = True

                manager._in_flight[first_uuid] = 1
                with (
                    patch("omnivoice.gpu_router.query_gpu_inventory", return_value=inventory),
                    patch("omnivoice.gpu_router.worker_health", side_effect=_health),
                    patch("omnivoice.gpu_router.send_spawn_worker_task", side_effect=_spawn) as mock_spawn,
                ):
                    worker = await manager.acquire_worker(require_slot=True)
                self.assertEqual(worker.uuid, second_uuid)
                self.assertTrue(spawned["requested"])
                mock_spawn.assert_called_once_with(second_uuid)
                await manager.release_worker(worker)
            finally:
                await manager.shutdown()

    async def test_acquire_worker_falls_back_to_single_worker_when_other_gpu_is_tight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            manager = OmniVoicePoolManager(config)
            await manager.startup()
            try:
                first_uuid, second_uuid = config.gpu_uuids
                first_port = config.worker_base_port
                inventory = {
                    first_uuid: GPUStat(2, first_uuid, "RTX 3060", 12288, 2500, 9100),
                    second_uuid: GPUStat(3, second_uuid, "RTX 3060", 12288, 5000, 6200),
                }

                def _health(port: int, timeout: float = 0.5):
                    if port == first_port:
                        return {"status": "ok"}
                    return None

                manager._in_flight[first_uuid] = 1

                async def _release_soon() -> None:
                    await asyncio.sleep(0.05)
                    await manager.release_worker(build_worker_specs(config)[0])

                release_task = asyncio.create_task(_release_soon())
                try:
                    with (
                        patch("omnivoice.gpu_router.query_gpu_inventory", return_value=inventory),
                        patch("omnivoice.gpu_router.worker_health", side_effect=_health),
                        patch("omnivoice.gpu_router.send_spawn_worker_task") as mock_spawn,
                    ):
                        worker = await manager.acquire_worker(require_slot=True)
                    self.assertEqual(worker.uuid, first_uuid)
                    mock_spawn.assert_not_called()
                    await manager.release_worker(worker)
                finally:
                    await release_task
            finally:
                await manager.shutdown()
