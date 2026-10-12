"""Install and validate the pinned PyTorch build without importing it here."""
import csv
import io
import json
import re
import shutil
import subprocess

VERSION = "2.8.0"
INDEX = "https://download.pytorch.org/whl/"

PROBE = r"""
import json
report = {}
try:
    import torch
    report.update(version=torch.__version__, cuda_version=torch.version.cuda,
                  cpu_usable=False, cuda_usable=False, cuda_busy=False)
    report["cpu_usable"] = torch.ones(2).sum().item() == 2
    if PROBE_CUDA and torch.cuda.is_available():
        try:
            value = torch.ones((2, 2), device="cuda", dtype=torch.bfloat16).float()
            value = torch.matmul(value, value)
            torch.cuda.synchronize()
            report["cuda_usable"] = value.sum().item() == 8
            report["gpu_name"] = torch.cuda.get_device_name()
        except torch.cuda.OutOfMemoryError:
            report["cuda_busy"] = True
            report["cuda_error"] = "GPU memory is temporarily unavailable"
        except Exception as exc:
            report["cuda_error"] = str(exc)
    elif PROBE_CUDA:
        report["cuda_error"] = "CUDA is not available to this Python environment"
except Exception as exc:
    report["error"] = str(exc)
print(json.dumps(report))
"""


def nvidia_devices():
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,driver_version,compute_cap",
             "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=10,
        )
        if result.returncode:
            return []
        devices = []
        for row in csv.reader(io.StringIO(result.stdout)):
            if len(row) != 3:
                continue
            name, driver, capability = (value.strip() for value in row)
            if not re.fullmatch(r"\d+(?:\.\d+)+", driver):
                continue
            try:
                capability = tuple(int(part) for part in capability.split("."))
            except ValueError:
                continue
            if len(capability) != 2:
                continue
            devices.append(dict(name=name, driver=driver, capability=capability))
        return devices
    except (OSError, subprocess.TimeoutExpired):
        return []


def preferred_build(requested, devices):
    if requested not in ("auto", "cpu", "cuda"):
        raise ValueError("Unknown installation device")
    if requested == "cpu":
        return "cpu", "CPU requested"
    for device in devices:
        driver = tuple(int(part) for part in device["driver"].split("."))
        capability = tuple(device["capability"])
        if driver >= (570, 26) and capability >= (7, 5):
            return "cu128", "NVIDIA GPU and driver support the CUDA 12.8 candidate"
    for device in devices:
        driver = tuple(int(part) for part in device["driver"].split("."))
        capability = tuple(device["capability"])
        # CUDA 12.x minor-version compatibility; the real kernel probe remains
        # mandatory. New architectures require the newer build above.
        if driver >= (525, 60, 13) and (5, 0) <= capability < (10, 0):
            return "cu126", "NVIDIA GPU and driver support the CUDA 12.6 candidate"
    return "cpu", "No supported NVIDIA GPU and driver detected"


def probe_torch(python, cuda=True):
    code = "PROBE_CUDA = " + repr(bool(cuda)) + "\n" + PROBE
    try:
        result = subprocess.run(
            [str(python), "-c", code], text=True, capture_output=True, timeout=30,
        )
        if result.returncode:
            return {"error": "PyTorch validation process failed"}
        report = json.loads(result.stdout)
        if not isinstance(report, dict):
            raise ValueError("Invalid probe result")
        return report
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return {"error": "PyTorch validation did not return a valid result in time"}


def pinned(probe):
    version = probe.get("version")
    return (isinstance(version, str) and version.split("+")[0] == VERSION
            and probe.get("cpu_usable") is True)


def gpu_ready(probe):
    return bool(probe.get("cuda_version") and
                (probe.get("cuda_usable") is True or probe.get("cuda_busy") is True))


def install_build(python, build):
    # The local version suffix forces CPU <-> CUDA replacement even when the
    # public version is unchanged. CUDA runtime dependencies are installed too.
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade",
         f"torch=={VERSION}+{build}", "--index-url", INDEX + build],
        check=True,
    )


def ensure_torch(python, requested="auto"):
    devices = nvidia_devices() if requested != "cpu" else []
    candidate, reason = preferred_build(requested, devices)
    probe = probe_torch(python, cuda=requested != "cpu")
    installed = False
    fallback = None
    if pinned(probe) and requested != "cpu" and gpu_ready(probe):
        # Preserve a proven compatible installation, including a working build
        # in containers where nvidia-smi is unavailable.
        candidate = "cuda"
        reason = ("Existing pinned CUDA build retained while GPU memory is busy"
                  if probe.get("cuda_busy") else "Existing pinned CUDA build passed validation")
    elif pinned(probe) and candidate == "cpu":
        # A CUDA wheel remains fully usable on CPU when no GPU is accessible.
        reason += "; existing pinned CPU execution passed validation"
    else:
        try:
            print(f"PyTorch: installing {candidate} ({reason}).", flush=True)
            install_build(python, candidate)
            installed = True
            probe = probe_torch(python, cuda=candidate != "cpu")
        except subprocess.CalledProcessError:
            if candidate == "cpu":
                raise
            fallback = "CUDA package installation failed"
        if candidate != "cpu" and (fallback or not pinned(probe) or not gpu_ready(probe)):
            fallback = fallback or probe.get("cuda_error") or probe.get("error") or "CUDA validation failed"
            print(f"PyTorch: {fallback}. Installing the CPU build.", flush=True)
            install_build(python, "cpu")
            installed = True
            candidate = "cpu"
            probe = probe_torch(python, cuda=False)
    if not pinned(probe):
        raise RuntimeError("Pinned PyTorch failed CPU validation; installation cannot continue")
    if candidate != "cpu" and not gpu_ready(probe):
        raise RuntimeError("CUDA build was selected without successful validation")
    selected = "cuda" if candidate != "cpu" and probe.get("cuda_usable") else "cpu"
    report = dict(requested=requested, selected_device=selected,
                  build=probe["version"], cuda_version=probe.get("cuda_version"),
                  cuda_usable=bool(probe.get("cuda_usable")),
                  cuda_busy=bool(probe.get("cuda_busy")), reason=reason,
                  fallback_reason=fallback, packages_changed=installed,
                  detected_gpus=devices)
    print(f"PyTorch: {report['build']}; validated device: {selected}.", flush=True)
    if report["cuda_busy"]:
        print("GPU memory is busy; CUDA support is retained for later use.", flush=True)
    return report
