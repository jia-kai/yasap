#!/usr/bin/env python3

from yasapctl.mount import MountController
from yasapctl.app import DEFAULT_CONFIG

from datetime import datetime
import pytz

def main():
    try:
        # Set up the mount controller with specific local time
        print("Setting up mount controller...")
        local_time = datetime.strptime(
            "Sat Apr 12 22:00:00 2025", "%a %b %d %H:%M:%S %Y")
        local_time = pytz.timezone('America/New_York').localize(local_time)

        mount = MountController.setup(**DEFAULT_CONFIG, time_local=local_time)

        print('LST', mount.local_sidereal())

        print(mount.get_state())
        mount.reset_home()

        current_ra, current_dec = mount.get_current_coord()
        print("Current position after reset: "
              f"RA={current_ra:.2f}°, DEC={current_dec:.2f}°")

        target_ra = current_ra + 30
        if target_ra >= 360:
            target_ra -= 360

        print("Rotating on RA axis to:"
              f" RA={target_ra:.2f}°, DEC={current_dec:.2f}°")
        mount.goto_coordinates(target_ra, current_dec, track=False)

        # Get the current coordinates after RA rotation
        current_ra, current_dec = mount.get_current_coord()
        print("Position after RA rotation:"
              f" RA={current_ra:.2f}°, DEC={current_dec:.2f}°")

        # Rotate 30 degrees on DEC axis
        target_dec = current_dec + 30.0
        if target_dec > 90.0:
            target_dec = 90.0 - (target_dec - 90.0) # Wrap around the pole

        print("Rotating 30 degrees on DEC axis to:"
              f" RA={current_ra:.2f}°, DEC={target_dec:.2f}°")
        mount.goto_coordinates(current_ra, target_dec, track=False)

        print("Test jitter")
        mount.jitter(100, -100)
        print("After jitter", mount.get_current_coord())

    finally:
        # Always ensure proper shutdown
        print("Shutting down mount controller...")
        MountController.shutdown()

if __name__ == "__main__":
    main()
