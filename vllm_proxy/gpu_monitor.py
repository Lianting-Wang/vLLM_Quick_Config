from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import psutil

try:
    import pynvml
except Exception:  # pragma: no cover - platform dependent
    pynvml = None


class GpuMonitor:
    def __init__(self) -> None:
        self.available = False
        self.error = ""
        if pynvml is None:
            self.error = "pynvml is not installed"
            return
        try:
            pynvml.nvmlInit()
            self.available = True
        except Exception as exc:  # pragma: no cover - platform dependent
            self.error = str(exc)

    def close(self) -> None:
        if self.available:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass

    @staticmethod
    def process_tree(pidfile: str) -> set[int]:
        if not pidfile:
            return set()
        try:
            pid = int(Path(pidfile).read_text(encoding="utf-8").strip())
            process = psutil.Process(pid)
            return {process.pid, *(child.pid for child in process.children(recursive=True))}
        except (OSError, ValueError, psutil.Error):
            return set()

    def snapshot(self, pidfile: str = "") -> dict[str, Any]:
        target_pids = self.process_tree(pidfile)
        result: dict[str, Any] = {
            "available": self.available,
            "error": self.error,
            "model_process_memory_mib": 0,
            "gpus": [],
        }
        if not self.available:
            return result
        try:
            count = pynvml.nvmlDeviceGetCount()
            model_total = 0
            for index in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                processes = []
                try:
                    processes.extend(pynvml.nvmlDeviceGetComputeRunningProcesses(handle))
                except Exception:
                    pass
                model_used = sum(int(p.usedGpuMemory) for p in processes if p.pid in target_pids and p.usedGpuMemory)
                model_total += model_used
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except Exception:
                    temperature = None
                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
                except Exception:
                    power_w = None
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                result["gpus"].append(
                    {
                        "index": index,
                        "name": name,
                        "used_mib": round(memory.used / 1024**2, 1),
                        "free_mib": round(memory.free / 1024**2, 1),
                        "total_mib": round(memory.total / 1024**2, 1),
                        "utilization_percent": int(utilization.gpu),
                        "temperature_c": temperature,
                        "power_w": round(power_w, 1) if power_w is not None else None,
                        "model_process_memory_mib": round(model_used / 1024**2, 1),
                    }
                )
            result["model_process_memory_mib"] = round(model_total / 1024**2, 1)
        except Exception as exc:  # pragma: no cover - platform dependent
            result["available"] = False
            result["error"] = str(exc)
        return result
