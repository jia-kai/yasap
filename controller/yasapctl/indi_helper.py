from .logging import LogManager

import PyIndi

from pathlib import Path
from typing import Type
import re
import shutil
import subprocess
import tempfile
import threading
import time

# Global configuration for retry behavior
RETRY_DELAY = 0.5  # Default delay between retries in seconds

class IndiServerManager:
    """Manages the INDI server process and log monitoring."""

    _logger: LogManager
    _driver: str
    _tmp_dir: Path

    _server_proc: subprocess.Popen
    _log_monitor_thread: threading.Thread
    _log_monitor_stop: threading.Event

    def __init__(self, driver: str, logger: LogManager):
        """Initialize the INDI server manager.

        Args:
            driver: The INDI driver to use
            logger: The logger to use for logging
        """
        self._logger = logger
        self._driver = driver
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="indi_"))
        self._logger.info(
            f"Created temporary directory for INDI server: {self._tmp_dir}")

        self._log_monitor_stop = threading.Event()

        self._start_server()
        self._start_log_monitor()

    def _start_server(self):
        """Start the INDI server."""
        self._logger.info(f"Starting INDI server with driver: {self._driver}")

        # Build the command to start the INDI server
        cmd = [
            "indiserver",
            "-l", str(self._tmp_dir),
            *self._driver.split()
        ]

        # Start the server process
        self._server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait a moment for the server to start
        time.sleep(0.5)

        # Check if the server is still running
        if self._server_proc.poll() is not None:
            stdout, stderr = self._server_proc.communicate()
            msg = f"INDI server failed to start: {stdout=} {stderr=}"
            self._logger.error(msg)
            raise RuntimeError(msg)

        self._logger.info("INDI server started successfully")

    def _start_log_monitor(self):
        """Start a background thread to monitor INDI log files."""
        self._log_monitor_stop.clear()
        self._log_monitor_thread = threading.Thread(
            target=self._monitor_log_files,
            daemon=True,
            name="INDI-Log-Monitor"
        )
        self._log_monitor_thread.start()
        self._logger.info("Started INDI log monitor thread")

    def _monitor_log_files(self):
        """Monitor INDI log files and forward their contents to the logger."""
        self._logger.info("INDI log monitor thread started")

        # Wait for log files to be created
        time.sleep(1)

        # Find the log file
        log_files = list(self._tmp_dir.glob("*.islog"))
        if not log_files:
            self._logger.warning(
                "No INDI log files found in temporary directory")
            return

        log_file = log_files[0]
        self._logger.info(f"Monitoring INDI log file: {log_file}")

        # Keep track of the last position in the file
        last_position = 0

        def check():
            nonlocal last_position
            # Open the file and seek to the last position
            with open(log_file, 'r') as f:
                f.seek(last_position)
                new_lines = f.readlines()

                # Update the last position
                last_position = f.tell()

                for line in new_lines:
                    line = line.strip()
                    if not line:
                        continue

                    # Parse the log line
                    # INDI log format: [YYYY-MM-DD HH:MM:SS] [LEVEL] Message
                    match = re.match(r'\[(.*?)\] \[(.*?)\] (.*)', line)
                    if match:
                        _, level, message = match.groups()

                        # Map INDI log levels to Python logging levels
                        if level == 'ERROR':
                            self._logger.error(f"[INDI] {message}")
                        elif level == 'WARNING':
                            self._logger.warning(f"[INDI] {message}")
                        elif level == 'INFO':
                            self._logger.info(f"[INDI] {message}")
                        else:
                            self._logger.debug(f"[INDI] {message}")
                    else:
                        # the line doesn't match
                        self._logger.warning(f"[INDI] {line}")

        try:
            while not self._log_monitor_stop.is_set():
                check()
                time.sleep(0.5)
        except Exception as e:
            self._logger.error(f"Error monitoring INDI log file: {e}")

    def shutdown(self):
        """Shutdown the INDI server and clean up resources."""
        # Stop the log monitor thread
        if hasattr(self, '_log_monitor_thread'):
            self._log_monitor_stop.set()
            self._log_monitor_thread.join(timeout=2)
            if self._log_monitor_thread.is_alive():
                self._logger.warning(
                    "INDI log monitor thread did not stop gracefully")

        # Terminate the server process
        if hasattr(self, '_server_proc'):
            self._logger.info("Terminating INDI server")
            self._server_proc.terminate()
            try:
                self._server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._logger.warning(
                    "INDI server did not terminate gracefully, forcing kill")
                self._server_proc.kill()

        # Move log files to log directory at CWD before cleanup
        if hasattr(self, '_tmp_dir') and self._tmp_dir.exists():
            # Create log directory if it doesn't exist
            log_dir = Path.cwd() / 'logs'
            log_dir.mkdir(exist_ok=True)

            # Find and move all log files
            log_files = list(self._tmp_dir.glob("*.islog"))
            for log_file in log_files:
                try:
                    # Create a timestamped filename to avoid overwriting
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    new_log_path = log_dir / f"{log_file.stem}_{timestamp}{log_file.suffix}"
                    shutil.copy2(log_file, new_log_path)
                    self._logger.info(f"Saved INDI log file to {new_log_path}")
                except Exception as e:
                    self._logger.error(f"Failed to save log file {log_file}: {e}")

            # Clean up the temporary directory
            shutil.rmtree(self._tmp_dir)

        self._logger.info("INDI server manager shutdown complete")

    def _log_and_raise(self, msg: str, exc_class: Type[Exception] = RuntimeError):
        """Log an error message and raise the specified exception with the same message.

        Args:
            msg: The error message to log and raise
            exc_class: The exception class to raise (default: RuntimeError)
        """
        self._logger.error(msg)
        raise exc_class(msg)

    def _wait_prop_ok(self, prop, prop_name: str, timeout: float = 5.0):
        """Wait for a property to become OK.

        Args:
            prop: The property to wait for
            prop_name: The name of the property (for logging)
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        start_time = time.time()
        max_retries = int(timeout / RETRY_DELAY)

        for attempt in range(max_retries):
            state = prop.getState()
            if state == PyIndi.IPS_OK:
                return
            if state != PyIndi.IPS_BUSY:
                break
            self._logger.debug(f"Waiting for {prop_name} to become OK (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        state_str = prop.getStateAsString()
        self._log_and_raise(f"{prop_name} did not become OK after {timeout} seconds. Current state: {state_str}")

    def chkSendNewSwitch(self, prop: PyIndi.PropertySwitch, timeout: float = 5.0):
        """Send a new switch property and wait for it to become OK.

        Args:
            prop: The switch property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewSwitch(prop)
        self._wait_prop_ok(prop, f"Switch property {prop.getName()}", timeout)

    def chkSendNewNumber(self, prop: PyIndi.PropertyNumber, timeout: float = 5.0):
        """Send a new number property and wait for it to become OK.

        Args:
            prop: The number property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewNumber(prop)
        self._wait_prop_ok(prop, f"Number property {prop.getName()}", timeout)

    def chkSendNewText(self, prop: PyIndi.PropertyText, timeout: float = 5.0):
        """Send a new text property and wait for it to become OK.

        Args:
            prop: The text property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewText(prop)
        self._wait_prop_ok(prop, f"Text property {prop.getName()}", timeout)

    def chkSendNewBlob(self, prop: PyIndi.PropertyBlob, timeout: float = 5.0):
        """Send a new blob property and wait for it to become OK.

        Args:
            prop: The blob property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewBlob(prop)
        self._wait_prop_ok(prop, f"Blob property {prop.getName()}", timeout)


class IndiDeviceEx:
    """Extended INDI device with helper methods for property access."""

    def __init__(self, device: PyIndi.BaseDevice, client: 'IndiClient'):
        """Initialize the extended device.

        Args:
            device: The INDI device
            client: The INDI client
        """
        self._device = device
        self._client = client
        self._logger = client.logger
        self._device_name = device.getDeviceName()

    def _log_and_raise(self, msg: str,
                       exc_class: Type[Exception] = RuntimeError):
        """Log an error message and raise the specified exception with the same message.

        Args:
            msg: The error message to log and raise
            exc_class: The exception class to raise (default: RuntimeError)
        """
        self._logger.error(msg)
        raise exc_class(msg)

    def getText(self, property_name: str, timeout: float = 5.0) -> PyIndi.PropertyText:
        """Get a text property with retries.

        Args:
            property_name: The name of the property
            timeout: Maximum time to wait in seconds

        Returns:
            The text property object

        Raises:
            RuntimeError: If the property is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            prop = self._device.getText(property_name)
            if prop:
                return prop

            self._logger.warning(f"Text property {property_name} not found (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(f"Text property {property_name} not found after {timeout} seconds")

    def getSwitch(self, property_name: str, timeout: float = 5.0) -> PyIndi.PropertySwitch:
        """Get a switch property with retries.

        Args:
            property_name: The name of the property
            timeout: Maximum time to wait in seconds

        Returns:
            The switch property object

        Raises:
            RuntimeError: If the property is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            prop = self._device.getSwitch(property_name)
            if prop:
                return prop

            self._logger.warning(f"Switch property {property_name} not found (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(f"Switch property {property_name} not found after {timeout} seconds")

    def getNumber(self, property_name: str, timeout: float = 5.0) -> PyIndi.PropertyNumber:
        """Get a number property with retries.

        Args:
            property_name: The name of the property
            timeout: Maximum time to wait in seconds

        Returns:
            The number property object

        Raises:
            RuntimeError: If the property is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            prop = self._device.getNumber(property_name)
            if prop:
                return prop

            self._logger.warning(f"Number property {property_name} not found (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(f"Number property {property_name} not found after {timeout} seconds")

    def getLight(self, property_name: str, timeout: float = 5.0) -> PyIndi.PropertyLight:
        """Get a light property with retries.

        Args:
            property_name: The name of the property
            timeout: Maximum time to wait in seconds

        Returns:
            The light property object

        Raises:
            RuntimeError: If the property is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            prop = self._device.getLight(property_name)
            if prop:
                return prop

            self._logger.warning(f"Light property {property_name} not found (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(f"Light property {property_name} not found after {timeout} seconds")

    def getBlob(self, property_name: str, timeout: float = 5.0) -> PyIndi.PropertyBlob:
        """Get a blob property with retries.

        Args:
            property_name: The name of the property
            timeout: Maximum time to wait in seconds

        Returns:
            The blob property object

        Raises:
            RuntimeError: If the property is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            prop = self._device.getBlob(property_name)
            if prop:
                return prop

            self._logger.warning(f"Blob property {property_name} not found (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(f"Blob property {property_name} not found after {timeout} seconds")

    def isConnected(self) -> bool:
        """Check if the device is connected.

        Returns:
            True if the device is connected, False otherwise
        """
        return self._device.isConnected()

    def getDeviceName(self) -> str:
        """Get the device name.

        Returns:
            The device name
        """
        return self._device_name


class IndiClient(PyIndi.BaseClient):
    """Client for communicating with INDI devices."""

    def __init__(self):
        super().__init__()
        self.logger = LogManager.get_instance()
        self.device_name = None
        self.connected = False

    def newDevice(self, baseDevice):
        """Called when a new device is created."""
        self.logger.info(f"New device detected: {baseDevice.getDeviceName()}")
        if self.device_name and baseDevice.getDeviceName() == self.device_name:
            self.logger.info(f"Found target device: {self.device_name}")

    def newProperty(self, property):
        """Called when a new property is created."""
        device_name = property.getDeviceName()
        property_name = property.getName()
        property_type = property.getTypeAsString()

        self.logger.info(f"New property: {property_name}"
                         f" ({property_type}) for device {device_name}")

    def _log_and_raise(self, msg: str, exc_class: Type[Exception] = RuntimeError):
        """Log an error message and raise the specified exception with the same message.

        Args:
            msg: The error message to log and raise
            exc_class: The exception class to raise (default: RuntimeError)
        """
        self.logger.error(msg)
        raise exc_class(msg)

    def _get_device(
            self, device_name: str, timeout: float = 5.0) -> PyIndi.BaseDevice:
        """Get a device by name with retries.

        Args:
            device_name: The name of the device to get
            timeout: Maximum time to wait in seconds

        Returns:
            The device object

        Raises:
            RuntimeError: If the device is not found within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY)
        for attempt in range(max_retries):
            device = self.getDevice(device_name)
            if device:
                return device

            self.logger.warning(
                f"Device {device_name} not found "
                f"(attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        self._log_and_raise(
            f"Device {device_name} not found after {timeout} seconds")

    def getDeviceEx(
            self, device_name: str,
            timeout: float = 5.0) -> IndiDeviceEx:
        """Get an extended device by name with retries.

        Args:
            device_name: The name of the device to get
            timeout: Maximum time to wait in seconds

        Returns:
            The extended device object

        Raises:
            RuntimeError: If the device is not found within the timeout
        """
        device = self._get_device(device_name, timeout)
        return IndiDeviceEx(device, self)

    def _wait_prop_ok(self, prop, prop_name: str, timeout: float = 5.0):
        """Wait for a property to become OK.

        Args:
            prop: The property to wait for
            prop_name: The name of the property (for logging)
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        max_retries = int(timeout / RETRY_DELAY) + 1
        for attempt in range(max_retries):
            state = prop.getState()
            if state == PyIndi.IPS_OK or state == PyIndi.IPS_IDLE:
                return
            if state != PyIndi.IPS_BUSY:
                break
            self.logger.debug(f"Waiting for {prop_name} to become OK (attempt {attempt+1}/{max_retries})")
            time.sleep(RETRY_DELAY)

        state_str = prop.getStateAsString()
        self._log_and_raise(f"{prop_name} did not become OK after {timeout} seconds. Current state: {state_str}")

    def chkSendNewSwitch(self, prop: PyIndi.PropertySwitch, timeout: float = 5.0):
        """Send a new switch property and wait for it to become OK.

        Args:
            prop: The switch property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewSwitch(prop)
        self._wait_prop_ok(prop, f"Switch property {prop.getName()}", timeout)

    def chkSendNewNumber(
            self, prop: PyIndi.PropertyNumber, timeout: float = 5.0):
        """Send a new number property and wait for it to become OK.

        Args:
            prop: The number property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewNumber(prop)
        self._wait_prop_ok(prop, f"Number property {prop.getName()}", timeout)

    def chkSendNewText(self, prop: PyIndi.PropertyText, timeout: float = 5.0):
        """Send a new text property and wait for it to become OK.

        Args:
            prop: The text property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewText(prop)
        self._wait_prop_ok(prop, f"Text property {prop.getName()}", timeout)

    def chkSendNewBlob(self, prop: PyIndi.PropertyBlob, timeout: float = 5.0):
        """Send a new blob property and wait for it to become OK.

        Args:
            prop: The blob property to send
            timeout: Maximum time to wait in seconds

        Raises:
            RuntimeError: If the property does not become OK within the timeout
        """
        self.sendNewBlob(prop)
        self._wait_prop_ok(prop, f"Blob property {prop.getName()}", timeout)
