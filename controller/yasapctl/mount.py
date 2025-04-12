from .logging import LogManager
from .indi_helper import IndiServerManager, IndiClient, IndiDeviceEx

from astropy.utils import iers
iers.conf.auto_download = False
from astropy.utils.iers import conf
conf.auto_max_age = None
from astropy.time import Time
from astropy.coordinates import EarthLocation
from astropy import units as u
import PyIndi

from datetime import datetime, timezone, timedelta
from typing import Optional, ClassVar, Type, Self
import time

class MountController:
    """Controller for mount operations."""

    _instance: ClassVar[Optional['MountController']] = None

    _client: IndiClient
    _logger: LogManager
    _device_name: str
    _driver: str
    _server_manager: IndiServerManager
    _device: IndiDeviceEx
    _time_local: datetime
    _latitude: float
    _longitude: float
    _elevation: float

    def __init__(self, latitude: float, longitude: float, elevation: float, mount_port: str,
                 device_name: str, driver: str,
                 time_local: Optional[datetime] = None):
        """Initialize the mount controller.

        Args:
            latitude: Mount latitude
            longitude: Mount longitude
            elevation: Mount elevation in meters
            mount_port: Port for mount connection
            device_name: INDI device name
            driver: INDI driver name
            time_local: Optional local time to set on the mount. If None, current local time is used.
        """

        if self._instance is not None:
            raise RuntimeError("MountController is a singleton")

        type(self)._instance = self
        self._logger = LogManager.get_instance()
        self._device_name = device_name
        self._driver = driver
        self._time_local = time_local or datetime.now()
        self._latitude = latitude
        self._longitude = longitude
        self._elevation = elevation

        # Create and start the INDI server manager
        self._server_manager = IndiServerManager(driver, self._logger)

        # Connect to the server
        self._connect(mount_port)

    def local_sidereal(self) -> float:
        """Compute the local sidereal time in hours.

        Returns:
            float: Local sidereal time in hours (0-24)
        """
        # Create an EarthLocation object with the mount's coordinates
        location = EarthLocation(
            lat=self._latitude * u.deg,
            lon=self._longitude * u.deg,
            height=self._elevation * u.m
        )

        # Create a Time object from the current local time
        time_obj = Time(self._time_local)

        # Calculate the local sidereal time
        lst = time_obj.sidereal_time('apparent', location)

        # Convert to hours (0-24)
        lst_hours = lst.hour
        return lst_hours

    def set_tracking(self, flag: bool, *, timeout: float=5.0):
        """set whether to be in tracking mode"""
        t = self._device.getSwitch('TELESCOPE_TRACK_STATE')
        if flag:
            t[0].setState(PyIndi.ISS_ON)
            t[1].setState(PyIndi.ISS_OFF)
        else:
            t[0].setState(PyIndi.ISS_OFF)
            t[1].setState(PyIndi.ISS_ON)
        self._client.chkSendNewSwitch(t, timeout=timeout)

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
        """Connect to the INDI server."""
        client = self._client = IndiClient()
        client.setServer('localhost', 7624)

        # Retry connection up to 5 times
        for attempt in range(5):
            if client.connectServer():
                self._logger.info("Connected to INDI server")
                break

            self._logger.warning(
                f"Failed to connect to INDI server (attempt {attempt+1}/5)")
            time.sleep(0.5)
        else:
            self._log_and_raise(
                "Failed to connect to INDI server after 5 attempts",
                ConnectionError)

        # Get the extended device
        device = self._device = client.getDeviceEx(self._device_name)

        # Set the device port before connecting
        self._logger.info(f"Setting device port to {mount_port}")
        if mount_port.startswith('/dev'):
            device_port = device.getText('DEVICE_PORT')
            device_port[0].setText(mount_port)
            client.chkSendNewText(device_port)
            self._logger.info(f"Device port set to {mount_port}")
        elif mount_port.startswith('tcp://'):
            host, port = mount_port[6:].split(':')
            mode = device.getSwitch('CONNECTION_MODE')
            assert mode[0].getName() == 'CONNECTION_SERIAL'
            mode[0].setState(PyIndi.ISS_OFF)
            mode[1].setState(PyIndi.ISS_ON)
            client.chkSendNewSwitch(mode)
            addr = device.getText('DEVICE_ADDRESS')
            addr[0].setText(host)
            addr[1].setText(port)
            client.chkSendNewText(addr)


        # Connect to the device if not already connected
        if not device.isConnected():
            self._logger.info(f"Connecting to device {self._device_name}...")
            connection = device.getSwitch('CONNECTION')
            connection[0].setState(PyIndi.ISS_ON)  # CONNECT
            connection[1].setState(PyIndi.ISS_OFF)  # DISCONNECT
            client.chkSendNewSwitch(connection)

        # iOptronV3 initialization is slow
        self.set_tracking(False, timeout=20.0)

        # Set the time on the mount
        self._setup_time()

        # Set the location on the mount
        self._setup_location()

        self._logger.info(
            f"Successfully connected to device {self._device_name}")

    def _setup_location(self):
        """Set the latitude, longitude, and elevation on the mount."""
        # Get the GEOGRAPHIC_COORD property
        geo_coord = self._device.getNumber('GEOGRAPHIC_COORD')

        # Set the latitude, longitude, and elevation
        geo_coord[0].setValue(self._latitude)  # Latitude
        geo_coord[1].setValue(self._longitude)  # Longitude
        geo_coord[2].setValue(self._elevation)  # Elevation

        self._logger.info(
            "Set mount location:"
            f" lat={self._latitude}, lon={self._longitude}, elev={self._elevation}")

        # Send the new coordinates
        self._client.chkSendNewNumber(geo_coord)

    def _setup_time(self):
        """Set the UTC time and UTC offset on the mount."""
        self._logger.info(f"attempt to set time to {self._time_local}")
        time_utc = self._device.getText('TIME_UTC')

        # Convert local time to UTC and format in ISO 8601
        utc_time = self._time_local.astimezone(timezone.utc)
        time_str = utc_time.strftime("%Y-%m-%dT%H:%M:%S")
        time_utc[0].setText(time_str)

        # Calculate UTC offset in hours (positive for timezones east of UTC)
        utc_offset = self._time_local.utcoffset()
        if utc_offset is None:
            utc_offset = timedelta(hours=0)  # Default to UTC if no offset
        offset_hours = utc_offset.total_seconds() / 3600
        time_utc[1].setText(f"{offset_hours:+.1f}")

        self._client.chkSendNewText(time_utc)
        self._logger.info(
            f"Set mount UTC time to {time_str} with UTC offset"
            f" {offset_hours:+.1f}")

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

        # Disconnect from the server
        if hasattr(instance, '_client'):
            logger.info("Disconnecting from INDI server")
            instance._client.disconnectServer()

        # Shutdown the server manager
        if hasattr(instance, '_server_manager') and instance._server_manager:
            logger.info("Shutting down INDI server manager")
            instance._server_manager.shutdown()

        # Reset the instance
        cls._instance = None
        logger.info("MountController shutdown complete")

    @classmethod
    def inst(cls) -> Optional['MountController']:
        """Get the global instance of MountController."""
        return cls._instance

    def goto_coordinates(self, ra: float, dec: float, track: bool = True):
        """Go to the specified coordinates."""

        # Set the ON_COORD_SET switch
        on_coord_set = self._device.getSwitch('ON_COORD_SET')

        # Configure tracking behavior
        on_coord_set[0].setState(
            PyIndi.ISS_ON if track else PyIndi.ISS_OFF)  # TRACK
        on_coord_set[1].setState(PyIndi.ISS_ON)  # SLEW
        on_coord_set[2].setState(PyIndi.ISS_OFF)  # SYNC

        # Send the new switch values
        self._client.chkSendNewSwitch(on_coord_set)

        # Set the coordinates
        radec = self._device.getNumber('EQUATORIAL_EOD_COORD')
        radec[0].setValue(ra)
        radec[1].setValue(dec)

        self._logger.info(
            f"Goto command sent: RA={ra}, DEC={dec}, Track={track}")

        # Send the new coordinates
        self._client.chkSendNewNumber(radec, timeout=120.0)

        self._logger.info("Goto completed successfully")

    def reset_home(self):
        """Set the mount's current position as its home position"""
        # Get the ON_COORD_SET switch
        home = self._device.getSwitch('TELESCOPE_HOME')
        home[0].setState(PyIndi.ISS_OFF)    # FIND
        home[1].setState(PyIndi.ISS_ON)     # SET
        home[2].setState(PyIndi.ISS_OFF)    # GO
        self._client.chkSendNewSwitch(home)

    def get_current_coord(self):
        """Get the current coordinates of the mount.

        Returns:
            tuple: A tuple containing (ra, dec) in degrees
        """
        # Get the current coordinates
        radec = self._device.getNumber('EQUATORIAL_EOD_COORD')

        # Extract RA and DEC values
        ra = radec[0].getValue()
        dec = radec[1].getValue()

        return (ra, dec)
