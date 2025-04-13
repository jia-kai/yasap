from .logging import LogManager
from .ioptron import MountClient, TcpComm, SerialComm, TimeInterface

#from astropy.utils import iers
#iers.conf.auto_download = False
#from astropy.utils.iers import conf
#conf.auto_max_age = None
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u

from datetime import datetime
from typing import Optional, ClassVar, Type, Self

class SystemTime(TimeInterface):
    """Time interface that uses the system time."""

    def get(self) -> datetime:
        """Get the current system time in the local timezone."""
        return datetime.now()

class MountController:
    """Controller for mount operations."""

    _instance: ClassVar[Optional['MountController']] = None

    _logger: LogManager
    _mount_client: MountClient
    _time_local: datetime
    _latitude: float
    _longitude: float
    _elevation: float

    def __init__(self, latitude: float, longitude: float, elevation: float, mount_port: str,
                 time_local: Optional[datetime] = None):
        """Initialize the mount controller.

        Args:
            latitude: Mount latitude
            longitude: Mount longitude
            elevation: Mount elevation in meters
            mount_port: Port for mount connection (e.g., 'tcp://10.10.100.254:8899' or '/dev/ttyUSB0')
            time_local: Optional local time to set on the mount. If None, current local time is used.
        """

        if self._instance is not None:
            raise RuntimeError("MountController is a singleton")

        type(self)._instance = self
        self._logger = LogManager.get_instance()
        self._time_local = time_local or datetime.now()
        self._latitude = latitude
        self._longitude = longitude
        self._elevation = elevation

        # Connect to the mount
        self._connect(mount_port)

    def local_sidereal(self) -> float:
        """Compute the local sidereal time in hours.

        Returns:
            float: Local sidereal time in hours (0-24)
        """
        # Create an EarthLocation object with the mount's coordinates
        location = EarthLocation(
            lat=self._latitude * u.deg,     # type: ignore
            lon=self._longitude * u.deg,    # type: ignore
            height=self._elevation * u.m    # type: ignore
        )

        # Create a Time object from the current local time
        time_obj = Time(self._time_local)

        # Calculate the local sidereal time
        lst = time_obj.sidereal_time('apparent', location)

        # Convert to hours (0-24)
        lst_hours = lst.hour
        return float(lst_hours) # type: ignore

    def set_tracking(self, flag: bool):
        """Set whether to be in tracking mode."""
        if flag:
            self._mount_client.start_1x_tracking()
        else:
            self._mount_client.stop_tracking()

    def _log_and_raise(self, msg: str,
                       exc_class: Type[Exception] = RuntimeError):
        """Log an error message and raise the specified exception with the same
        message.

        Args:
            msg: The error message to log and raise
            exc_class: The exception class to raise (default: RuntimeError)
        """
        self._logger.error(msg)
        raise exc_class(msg)

    def _connect(self, mount_port: str):
        """Connect to the mount."""
        self._logger.info(f"Connecting to mount at {mount_port}")

        # Create the appropriate communication interface
        if mount_port.startswith('tcp://'):
            # Parse TCP connection string
            host, port = mount_port[6:].split(':')
            comm = TcpComm(host, int(port))
        else:
            # Assume it's a serial port
            comm = SerialComm(mount_port)

        # Create the time interface
        time_interface = SystemTime()

        # Create the mount client
        self._mount_client = MountClient(
            comm=comm,
            time=time_interface,
            longitude=self._longitude,
            latitude=self._latitude
        )

        self._logger.info("Successfully connected to mount")

    @classmethod
    def setup(cls, **kwargs) -> Self:
        """Set up the mount controller."""
        if cls._instance is not None:
            raise RuntimeError("MountController is already set up")
        cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def shutdown(cls):
        """Shutdown the mount controller."""
        if cls._instance is None:
            return

        instance = cls._instance
        logger = instance._logger

        # Close the mount connection
        if hasattr(instance, '_mount_client'):
            logger.info("Closing mount connection")
            instance._mount_client.comm.close()

        # Reset the instance
        cls._instance = None
        logger.info("MountController shutdown complete")

    @classmethod
    def inst(cls) -> Optional['MountController']:
        """Get the global instance of MountController."""
        return cls._instance

    def goto_coordinates(self, ra: float, dec: float, track: bool = True):
        """Go to the specified coordinates."""
        self._logger.info(f"Goto command sent: RA={ra}, DEC={dec}, Track={track}")

        # Go to the coordinates
        self._mount_client.goto(ra, dec)

        if track:
            self._mount_client.start_1x_tracking()
        else:
            self._mount_client.stop_tracking()

        self._logger.info("Goto completed successfully")

    def reset_home(self):
        """Set the mount's current position as its home position."""
        #self._mount_client.reset_coord(180, 0)

    def get_current_coord(self):
        """Get the current coordinates of the mount.

        Returns:
            tuple: A tuple containing (ra, dec) in degrees
        """
        return self._mount_client.get_coord()

    def get_state(self):
        """Get the current state of the mount.

        Returns:
            SystemState: The current state of the mount
        """
        return self._mount_client.get_state()

    def jitter(self, ra_ms: int, dec_ms: int):
        self._mount_client.jitter(ra_ms, dec_ms)
