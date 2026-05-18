import logging
import os
import random
import signal
import socket
import subprocess
import threading
import time
import uuid
from pathlib import Path

import requests

from desktop_env.providers.base import Provider

logger = logging.getLogger("desktopenv.providers.nvcf_singularity.NVCFSingularityProvider")
logger.setLevel(logging.INFO)

RETRY_INTERVAL = 10
WAIT_TIME = 3
DEFAULT_SIF_PATH = "/lustre/fsw/portfolios/nvr/users/mingjiel/workspace/nvcf-osworld-eval/osworld-linux.sif"

# Port ranges for direct singularity mode
API_PORT_RANGE = (15000, 19999)
VNC_PORT_RANGE = (18000, 22999)
CHROME_PORT_RANGE = (19000, 22999)
VLC_PORT_RANGE = (20000, 22999)

# Env-manager mode settings
DEFAULT_SIF_NAME = "kasm-ubuntu-noble-gnome-osworld"
ENV_READY_TIMEOUT = 600
ENV_STATUS_POLL_INTERVAL = 5


class PortAllocationError(Exception):
    pass


class NVCFSingularityProvider(Provider):
    """
    Singularity-based provider with two operating modes:

    1. Direct mode (default): Runs a .sif container directly via `singularity run`.
       Activated when APPTAINER_ENV_MANAGER_URL is NOT set.

    2. Env-manager mode: Delegates lifecycle to a local apptainer-env-manager
       service that handles SIF download, extraction, chroot, and VNC setup.
       Activated when APPTAINER_ENV_MANAGER_URL is set.

    In both modes, the DesktopEnv controller talks directly to per-environment
    ports for screenshot/execute/screen_size.

    Env vars (env-manager mode):
        APPTAINER_ENV_MANAGER_URL  - env-manager base URL (enables env-manager mode)
        APPTAINER_SIF_NAME         - SIF image name to launch (default: kasm-ubuntu-noble-gnome-osworld)

    Env vars (direct mode):
        NVCF_SINGULARITY_SIF_PATH  - path to .sif file (default: see DEFAULT_SIF_PATH)
    """

    _port_allocation_lock = threading.Lock()

    def __init__(self, region: str = None):
        super().__init__(region)
        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None

        # Env-manager mode state
        self._env_manager_url = os.environ.get("APPTAINER_ENV_MANAGER_URL")
        if self._env_manager_url:
            self._env_manager_url = self._env_manager_url.rstrip("/")
        self._env_id = None
        self._sif_name = None  # set during _start_via_env_manager

        # Direct singularity mode state
        self.process: subprocess.Popen = None
        self.process_pid: int = None
        self._stdout_fh = None
        self._stderr_fh = None

        if not self._env_manager_url:
            self._check_singularity_availability()

    @property
    def _use_env_manager(self) -> bool:
        return self._env_manager_url is not None

    # ------------------------------------------------------------------
    # Shared
    # ------------------------------------------------------------------

    def _wait_for_vm_ready(self, timeout: int = 300):
        """Wait for the per-env Flask API to serve screenshots."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"http://localhost:{self.server_port}/screenshot",
                    timeout=(10, 10),
                )
                if response.status_code == 200:
                    return True
            except Exception:
                pass

            # In direct mode, check if the process crashed
            if not self._use_env_manager and self.process and self.process.poll() is not None:
                self._read_and_raise_error()

            logger.info("Checking if nvcf_singularity container is ready...")
            time.sleep(RETRY_INTERVAL)

        raise TimeoutError("nvcf_singularity failed to become ready within timeout period")

    def get_ip_address(self, path_to_vm: str) -> str:
        if not all([self.server_port, self.chromium_port, self.vnc_port, self.vlc_port]):
            raise RuntimeError("Container not started - ports not allocated")
        return f"localhost:{self.server_port}:{self.chromium_port}:{self.vnc_port}:{self.vlc_port}"

    def save_state(self, path_to_vm: str, snapshot_name: str):
        raise NotImplementedError("Snapshots not available for nvcf_singularity provider")

    def revert_to_snapshot(self, path_to_vm: str, snapshot_name: str):
        self.stop_emulator(path_to_vm)

    def start_emulator(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        if self._use_env_manager:
            self._start_via_env_manager()
        else:
            self._start_via_singularity(path_to_vm, headless, os_type)

    def stop_emulator(self, path_to_vm: str, region=None, *args, **kwargs):
        del path_to_vm, region, args, kwargs
        if self._use_env_manager:
            self._stop_via_env_manager()
        else:
            self._stop_via_singularity()

    # ------------------------------------------------------------------
    # Env-manager mode
    # ------------------------------------------------------------------

    def _start_via_env_manager(self):
        # Support SIF diversity: if APPTAINER_SIF_NAMES is set (comma-separated),
        # randomly select one per launch for visual/env diversity in trajectories.
        sif_names_env = os.environ.get("APPTAINER_SIF_NAMES", "")
        if sif_names_env:
            _sif_list = [s.strip() for s in sif_names_env.split(",") if s.strip()]
            sif_name = random.choice(_sif_list)
        else:
            sif_name = os.environ.get("APPTAINER_SIF_NAME", DEFAULT_SIF_NAME)
        self._sif_name = sif_name  # expose for metadata tracking
        self._env_id = f"env-{uuid.uuid4().hex[:12]}"

        logger.info(
            "Launching environment %s (sif=%s) via env-manager at %s",
            self._env_id, sif_name, self._env_manager_url,
        )

        try:
            resp = requests.post(
                f"{self._env_manager_url}/env/launch",
                json={"sif_name": sif_name, "env_id": self._env_id},
                timeout=30,
            )
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(
                f"Failed to launch environment via env-manager: {e}"
            ) from e

        try:
            self._wait_for_env_ready()
            self._wait_for_vm_ready()
        except Exception:
            self.stop_emulator("")
            raise

    def _wait_for_env_ready(self, timeout: int = ENV_READY_TIMEOUT):
        """Poll the env-manager until the environment status is 'ready'."""
        start = time.time()
        last_status = None

        while time.time() - start < timeout:
            try:
                resp = requests.get(
                    f"{self._env_manager_url}/env/{self._env_id}/status",
                    timeout=10,
                )
                resp.raise_for_status()
                data = resp.json()
                status = data.get("status")

                if status != last_status:
                    logger.info(
                        "Environment %s status: %s (%.0fs)",
                        self._env_id, status, time.time() - start,
                    )
                    last_status = status

                if status == "ready":
                    ports = data.get("ports", {})
                    self.server_port = ports.get("api")
                    self.vnc_port = ports.get("vnc")
                    self.chromium_port = ports.get("chrome")
                    self.vlc_port = ports.get("vlc")

                    if not all([self.server_port, self.vnc_port,
                                self.chromium_port, self.vlc_port]):
                        raise RuntimeError(
                            f"Environment ready but missing ports: {ports}"
                        )

                    logger.info(
                        "Environment %s ready — API:%d VNC:%d Chrome:%d VLC:%d",
                        self._env_id, self.server_port, self.vnc_port,
                        self.chromium_port, self.vlc_port,
                    )
                    return

                if status == "error":
                    error_msg = data.get("error_msg", "unknown error")
                    raise RuntimeError(
                        f"Environment {self._env_id} entered error state: {error_msg}"
                    )

            except requests.RequestException as e:
                logger.warning("Failed to poll env status: %s", e)

            time.sleep(ENV_STATUS_POLL_INTERVAL)

        raise TimeoutError(
            f"Environment {self._env_id} did not become ready within {timeout}s "
            f"(last status: {last_status})"
        )

    def _stop_via_env_manager(self):
        if self._env_id:
            logger.info("Destroying environment %s via env-manager", self._env_id)
            try:
                resp = requests.delete(
                    f"{self._env_manager_url}/env/{self._env_id}",
                    timeout=30,
                )
                if resp.status_code == 404:
                    logger.warning("Environment %s already gone", self._env_id)
                else:
                    resp.raise_for_status()
                    logger.info("Environment %s destroyed", self._env_id)
            except requests.RequestException as e:
                logger.error("Failed to destroy environment %s: %s", self._env_id, e)

        self._env_id = None
        self.server_port = None
        self.chromium_port = None
        self.vnc_port = None
        self.vlc_port = None

    # ------------------------------------------------------------------
    # Direct singularity mode (original behavior)
    # ------------------------------------------------------------------

    @staticmethod
    def _check_singularity_availability():
        try:
            subprocess.run(
                ['singularity', '--version'],
                capture_output=True, text=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                'Singularity/Apptainer is not available. '
                'Please install it to use NVCFSingularityProvider.'
            ) from e

    @staticmethod
    def _check_port_available(port: int) -> bool:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False
        finally:
            sock.close()

    def _find_available_port(self, min_port: int, max_port: int, max_attempts: int = 50) -> int:
        rng = random.SystemRandom()
        ports = list(range(min_port, max_port + 1))
        rng.shuffle(ports)

        for port in ports[:max_attempts]:
            if self._check_port_available(port):
                return port
        raise PortAllocationError(f"No available ports found in range {min_port}-{max_port}")

    def _allocate_ports(self) -> tuple:
        """Allocate unique ports for API, VNC, Chrome, and VLC."""
        with NVCFSingularityProvider._port_allocation_lock:
            api_port = self._find_available_port(*API_PORT_RANGE)
            vnc_port = self._find_available_port(*VNC_PORT_RANGE)
            chrome_port = self._find_available_port(*CHROME_PORT_RANGE)
            vlc_port = self._find_available_port(*VLC_PORT_RANGE)
            return api_port, vnc_port, chrome_port, vlc_port

    def _read_and_raise_error(self):
        self._close_log_handles()
        log_dir = Path("/tmp/osworld_nvcf_singularity_logs")
        err_files = sorted(log_dir.glob("singularity_*.err"), reverse=True)
        error_output = ""
        if err_files:
            error_output = err_files[0].read_text()
        raise RuntimeError(
            f"Singularity container exited unexpectedly. "
            f"Return code: {self.process.returncode}\nError: {error_output}"
        )

    def _start_via_singularity(self, path_to_vm: str, headless: bool, os_type: str = "Ubuntu"):
        del path_to_vm, headless, os_type

        sif_path = os.environ.get("NVCF_SINGULARITY_SIF_PATH", DEFAULT_SIF_PATH)
        if not os.path.exists(sif_path):
            raise FileNotFoundError(
                f"Singularity image not found: {sif_path}. "
                f"Please build or download the .sif image first."
            )

        try:
            self.server_port, self.vnc_port, self.chromium_port, self.vlc_port = self._allocate_ports()

            logger.info(
                "Allocated ports - API: %d, VNC: %d, Chrome: %d, VLC: %d",
                self.server_port, self.vnc_port, self.chromium_port, self.vlc_port,
            )

            cmd = [
                'singularity', 'run',
                '--contain',
                '--cleanenv',
                '--pid',
                '--writable-tmpfs',
                '--no-mount', 'home,cwd,tmp',
                '--home', '/home/user',
                '--env', f'API_PORT={self.server_port}',
                '--env', f'VNC_PORT={self.vnc_port}',
                '--env', f'CHROME_PORT={self.chromium_port}',
                '--env', f'VLC_PORT={self.vlc_port}',
                sif_path,
            ]

            log_dir = Path("/tmp/osworld_nvcf_singularity_logs")
            log_dir.mkdir(parents=True, exist_ok=True)

            timestamp = int(time.time())
            stdout_path = log_dir / f"singularity_{timestamp}.out"
            stderr_path = log_dir / f"singularity_{timestamp}.err"

            self._stdout_fh = open(stdout_path, 'w')
            self._stderr_fh = open(stderr_path, 'w')

            self.process = subprocess.Popen(
                cmd,
                stdout=self._stdout_fh,
                stderr=self._stderr_fh,
                text=True,
                start_new_session=True,
            )
            self.process_pid = self.process.pid

            time.sleep(2)
            if self.process.poll() is not None:
                self._close_log_handles()
                with open(stderr_path, 'r') as f:
                    error_output = f.read()
                raise RuntimeError(
                    f"Singularity container failed to start. "
                    f"Return code: {self.process.returncode}\nError: {error_output}"
                )

            logger.info("Singularity process started with PID: %d", self.process_pid)
            logger.info("Logs: stdout=%s, stderr=%s", stdout_path, stderr_path)

            logger.info(
                "Started nvcf_singularity with image '%s' "
                "(api=%d, vnc=%d, chrome=%d, vlc=%d)",
                sif_path,
                self.server_port, self.vnc_port, self.chromium_port, self.vlc_port,
            )
            self._wait_for_vm_ready()
        except Exception:
            self.stop_emulator("")
            raise

    def _close_log_handles(self):
        for fh in (self._stdout_fh, self._stderr_fh):
            if fh:
                try:
                    fh.close()
                except Exception:
                    pass
        self._stdout_fh = None
        self._stderr_fh = None

    def _stop_via_singularity(self):
        try:
            self._close_log_handles()

            if self.process_pid is not None:
                logger.info("Stopping Singularity process (PID: %d)", self.process_pid)
                try:
                    os.kill(self.process_pid, signal.SIGTERM)
                    time.sleep(2)
                    try:
                        os.kill(self.process_pid, 0)
                        os.kill(self.process_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                except ProcessLookupError:
                    pass
                except Exception as e:
                    logger.warning("Failed to kill Singularity process: %s", e)

            time.sleep(WAIT_TIME)
        except Exception as e:
            logger.error("Error stopping nvcf_singularity container: %s", e)
        finally:
            self.process = None
            self.process_pid = None
            self.server_port = None
            self.chromium_port = None
            self.vnc_port = None
            self.vlc_port = None
