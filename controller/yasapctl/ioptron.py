# iOptron mount driver

from .logging import LogManager

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
import datetime
import socket
import time
import typing

# Exception classes for communication errors.
class CommunicationError(Exception):
    pass

class TimeoutError(CommunicationError):
    pass

class CommInterface(ABC):
    """abstract Communication Interface"""
    def __init__(self, timeout: float):
        self.timeout = timeout  # seconds

    @abstractmethod
    def send(self, cmd: str) -> None:
        """Send an ASCII command to the mount."""
        pass

    @abstractmethod
    def recv(self, use_delim: bool, fix_len: typing.Optional[int]=None) -> str:
        """Receive response from the mount. Raise TimeoutError if timeout.

        :param use_delim: if True, read until `#` is sent; if False, read only
            one character
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the communication interface."""
        pass


# TCP communication interface.
class TcpComm(CommInterface):
    def __init__(self, host: str, port: int, timeout: float = 5.0):
        super().__init__(timeout)
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)

    def send(self, cmd: str) -> None:
        try:
            self.sock.sendall(cmd.encode('ascii'))
        except socket.timeout:
            raise TimeoutError("TCP send timeout")
        except Exception as e:
            raise CommunicationError(f"TCP send error: {e}")

    def recv(self, use_delim: bool, fix_len: typing.Optional[int]=None) -> str:
        end_marker = b'#'
        start = time.time()
        result = b''

        while True:
            try:
                if fix_len is not None:
                    # Read exactly the remaining bytes needed
                    remaining = fix_len - len(result)
                    if remaining <= 0:
                        break
                    chunk = self.sock.recv(remaining)
                    if not chunk:
                        raise TimeoutError("TCP recv timeout")
                    result += chunk
                    if len(result) >= fix_len:
                        break
                elif use_delim:
                    chunk = self.sock.recv(1024)
                    if not chunk:
                        break
                    result += chunk
                    if end_marker in chunk:
                        break
                else:
                    chunk = self.sock.recv(1)
                    if not chunk:
                        raise TimeoutError("TCP recv timeout")
                    result += chunk
                    break
            except socket.timeout:
                raise TimeoutError("TCP recv timeout")
            if time.time() - start > self.timeout:
                raise TimeoutError("TCP recv overall timeout")

        return result.decode('ascii')

    def close(self) -> None:
        """Close the TCP connection."""
        if hasattr(self, 'sock'):
            self.sock.close()


class SerialComm(CommInterface):
    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 5.0):
        super().__init__(timeout)
        import serial  # pyserial must be installed
        try:
            self.ser = serial.Serial(port=port,
                                     baudrate=baudrate,
                                     timeout=timeout,
                                     parity=serial.PARITY_NONE,
                                     bytesize=serial.EIGHTBITS,
                                     stopbits=serial.STOPBITS_ONE)
        except Exception as e:
            raise CommunicationError(f"Serial port error: {e}")

    def send(self, cmd: str) -> None:
        try:
            self.ser.write(cmd.encode('ascii'))
        except Exception as e:
            raise CommunicationError(f"Serial send error: {e}")

    def recv(self, use_delim: bool, fix_len: typing.Optional[int]=None) -> str:
        """Receive response from the mount. Raise TimeoutError if timeout.

        :param use_delim: if True, read until `#` is sent; if False, read only
            one character
        :param fix_len: if provided, read exactly this many bytes
        """
        end_marker = b'#'
        start = time.time()
        result = b''

        while True:
            if fix_len is not None:
                # Read exactly the remaining bytes needed
                remaining = fix_len - len(result)
                if remaining <= 0:
                    break
                data = self.ser.read(remaining)
                if not data:
                    raise TimeoutError("Serial recv timeout")
                result += data
                if len(result) >= fix_len:
                    break
            elif use_delim:
                byte = self.ser.read(1)
                if not byte:
                    raise TimeoutError("Serial recv timeout")
                result += byte
                if byte == end_marker:
                    break
            else:
                byte = self.ser.read(1)
                if not byte:
                    raise TimeoutError("Serial recv timeout")
                result += byte
                break
            if time.time() - start > self.timeout:
                raise TimeoutError("Serial recv overall timeout")

        return result.decode('ascii')

    def close(self) -> None:
        """Close the serial connection."""
        if hasattr(self, 'ser'):
            self.ser.close()


class TimeInterface(ABC):
    @abstractmethod
    def get(self) -> datetime.datetime:
        pass

class GPSState(Enum):
    NO_GPS = 0
    NO_DATA = 1
    WORKING = 2

class MountState(Enum):
    STOP_NZ = 0
    TRACKING_NO_PEC = 1
    SLEWING = 2
    AUTOGUIDING = 3
    MERIDIAN_FLIPPING = 4
    TRACKING_PEC = 5
    PARKED = 6
    STOP_HOME = 7

class TrackingRate(Enum):
    SIDEREAL = 0
    LUNAR = 1
    SOLAR = 2
    KING = 3
    CUSTOM = 4

class TimeSource(Enum):
    UNKNOWN = 0
    RS232 = 1
    HC = 2
    GPS = 3

@dataclass(slots=True, frozen=True)
class SystemState:
    longitude: float
    latitude: float
    gps: GPSState
    state: MountState
    tracking_rate: TrackingRate
    arrow_speed: int
    time_source: TimeSource
    is_north: bool

class MountClient:
    comm: CommInterface
    time: TimeInterface
    longitude: float
    latitude: float
    northern: bool

    def __init__(self,
                 comm: CommInterface, time: TimeInterface,
                 longitude: float, latitude: float):
        """
        :param comm: A communication interface instance (TCP or Serial).
        :param longitude: In degrees (positive for East).
        :param latitude: In degrees (positive for North).
        """
        self.comm = comm
        self.time = time
        self.logger = LogManager.get_instance()
        self.mount_model = None
        self.longitude = longitude
        self.latitude = latitude
        # Infer hemisphere from latitude (>= 0 is Northern).
        self.northern = (latitude >= 0)

        self._initialize_mount()

    @staticmethod
    def _compute_j2000_ms(dt: datetime.datetime) -> int:
        """Compute milliseconds since J2000 epoch from a datetime.

        The J2000 epoch is January 1, 2000, 12:00:00 TT (Terrestrial Time). This
        implementation uses UTC for simplicity, which introduces a small error
        (about 32.184 seconds) but is acceptable for mount control purposes.

        Args:
            dt: The datetime to convert

        Returns:
            int: Milliseconds since J2000 epoch
        """

        j2000 = datetime.datetime(2000, 1, 1, 11, 58, 55, 816,
                                  tzinfo=datetime.timezone.utc)

        # Convert input datetime to UTC if it has timezone info
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc)
        else:
            dt = dt.replace(tzinfo=datetime.timezone.utc)

        # Calculate time difference in milliseconds
        delta = dt - j2000
        return int(delta.total_seconds() * 1000)

    @staticmethod
    def format_signed_int(val: int, digits: int):
        assert isinstance(val, int)
        if val > 0:
            sign = '+'
        else:
            sign = '-'
            val = -val
        vs = str(val)
        assert len(vs) <= digits, (val, digits)
        vs = '0' * (digits - len(vs)) + vs
        return sign + vs

    @classmethod
    def format_longlat(cls, deg: float) -> str:
        assert -180 <= deg <= 180, deg
        return cls.format_signed_int(int(round(deg * 3600 * 100)), 8)

    @staticmethod
    def format_ra(ra_deg: float) -> str:
        assert 0 <= ra_deg < 360
        val = int(round(ra_deg * 3600 * 100))
        return f"{val:09d}"

    @staticmethod
    def format_dec(dec_deg: float) -> str:
        assert -90 <= dec_deg <= 90
        sign = '+' if dec_deg >= 0 else '-'
        val = int(round(abs(dec_deg) * 3600 * 100))
        return f"{sign}{val:08d}"

    def send_cmd(self, cmd: str, *,
                 use_delim: bool = True, wait_resp: bool = True,
                 fix_len: typing.Optional[int]=None,
                 ) -> str:
        """Sends a command and returns its response

        :param use_delim: see :meth:`CommInterface.recv`
        :param wait_resp: whether to wait for response; if False, return empty
            string
        """
        self.logger.debug(f">= {cmd}")
        self.comm.send(cmd)
        if not wait_resp:
            assert fix_len is None
            return ''
        resp = self.comm.recv(use_delim=use_delim, fix_len=fix_len)
        self.logger.debug(f"<= {resp}")
        if not resp:
            raise CommunicationError("Empty response received")
        return resp.strip()

    def check_cmd(self, cmd: str):
        """send the command and assert that the response is '1'"""
        ret = self.send_cmd(cmd, use_delim=False)
        if ret != '1':
            raise CommunicationError(f"expected 1, got {ret}")

    def _initialize_mount(self):
        self.logger.info("Starting mount initialization sequence")
        # Get mount model.
        resp = self.send_cmd(":MountInfo#", fix_len=4)
        self.mount_model = resp  # Can be mapped to a model name if needed.
        self.logger.info(f"Mount model: {self.mount_model}")

        self.send_cmd(":MountInfo#", fix_len=4)
        self.send_cmd(":FW1#")
        self.send_cmd(":FW2#")

        current_time = self.time.get()
        ms_since_j2000 = self._compute_j2000_ms(current_time)
        self.check_cmd(f":SUT{ms_since_j2000:013d}#")

        utc_offset_dt = current_time.utcoffset()
        if utc_offset_dt is None:
            utc_offset_minutes =0
        else:
            utc_offset_minutes = int(utc_offset_dt.total_seconds() / 60)
        self.check_cmd(f":SG{self.format_signed_int(utc_offset_minutes, 3)}#")

        self.check_cmd(f':SLO{self.format_longlat(self.longitude)}#')
        self.check_cmd(f':SLA{self.format_longlat(self.latitude)}#')

        hemisphere_cmd = ":SHE1#" if self.northern else ":SHE0#"
        self.check_cmd(hemisphere_cmd)

    def _prepare_coord(self, ra_deg: float, dec_deg: float):
        ra_str = self.format_ra(ra_deg)
        dec_str = self.format_dec(dec_deg)
        self.check_cmd(f":SRA{ra_str}#")
        self.check_cmd(f":Sd{dec_str}#")

    def reset_coord(self, ra_deg: float, dec_deg: float):
        self.logger.info(
            f"Reset coordinates to RA: {ra_deg}deg, Dec: {dec_deg}deg")
        self._prepare_coord(ra_deg, dec_deg)
        self.check_cmd(":CM#")

    def get_coord(self) -> tuple[float, float]:
        resp = self.send_cmd(":GEP#")
        if len(resp) != 21:
            raise CommunicationError(f"Invalid response from :GEP#: {resp}")
        dec_val = int(resp[:9])
        ra_val = int(resp[9:18])
        dec_deg = dec_val / (3600 * 100)
        ra_deg = ra_val / (3600 * 100)
        return (ra_deg, dec_deg)

    def goto(self, ra_deg: float, dec_deg: float):
        self.logger.info(f"Slewing to RA: {ra_deg}deg, Dec: {dec_deg}deg")
        self._prepare_coord(ra_deg, dec_deg)
        self.check_cmd(":MS1#")
        while self.get_state().state == MountState.SLEWING:
            time.sleep(.5)

    def start_1x_tracking(self):
        self.logger.info("Starting sidereal tracking")
        self.check_cmd(":RT0#")
        self.check_cmd(":ST1#")

    def stop_tracking(self):
        self.logger.info("Stopping tracking")
        self.check_cmd(":ST0#")

    def jitter(self, delta_ra_ms: int, delta_dec_ms: int):
        self.logger.info(
            f"Applying jitter: RA delta {delta_ra_ms} ms,"
            f" Dec delta {delta_dec_ms} ms")
        send = lambda c: self.send_cmd(c, wait_resp=False)
        if delta_ra_ms > 0:
            cmd = f":ZS{delta_ra_ms:05d}#"
            send(cmd)
        elif delta_ra_ms < 0:
            cmd = f":ZQ{-delta_ra_ms:05d}#"
            send(cmd)
        if delta_dec_ms > 0:
            cmd = f":ZE{delta_dec_ms:05d}#"
            send(cmd)
        elif delta_dec_ms < 0:
            cmd = f":ZC{-delta_dec_ms:05d}#"
            send(cmd)

    def get_state(self) -> SystemState:
        """get the current state of the mount using the GLS command."""
        resp = self.send_cmd(":GLS#")
        if len(resp) != 24:
            raise CommunicationError(f"Invalid response from :GLS#: {resp}")

        return SystemState(
            longitude=int(resp[:9]) / (3600 * 100),
            latitude=int(resp[9:17]) / (3600 * 100) - 90,
            gps=GPSState(int(resp[17])),
            state=MountState(int(resp[18])),
            tracking_rate=TrackingRate(int(resp[19])),
            arrow_speed=int(resp[20]),
            time_source=TimeSource(int(resp[21])),
            is_north=int(resp[22]) == 1
        )
