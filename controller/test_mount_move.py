#!/usr/bin/env python3

from yasapctl.mount import MountController
from yasapctl.app import DEFAULT_CONFIG

from datetime import datetime
import pytz
import time

def main():
    try:
        # Set up the mount controller with specific local time
        print("Setting up mount controller...")
        local_time = datetime.strptime("Sat Apr 12 22:00:00 2025", "%a %b %d %H:%M:%S %Y")
        local_time = pytz.timezone('America/New_York').localize(local_time)

        mount = MountController.setup(**DEFAULT_CONFIG, time_local=local_time)

        print('LST', mount.local_sidereal())

        # Reset the mount's zero position
        print("Resetting mount zero position...")
        time.sleep(3)
        current_ra, current_dec = mount.get_current_coord()
        print(f"Current position before reset: RA={current_ra:.2f}°, DEC={current_dec:.2f}°")
        mount.reset_home()

        time.sleep(3)
        # Get the current coordinates after reset
        current_ra, current_dec = mount.get_current_coord()
        print(f"Current position after reset: RA={current_ra:.2f}°, DEC={current_dec:.2f}°")

        target_ra = current_ra + 90 / 360 * 24
        if target_ra >= 24.0:
            target_ra -= 24.0

        print(f"Rotating on RA axis to: RA={target_ra:.2f}°, DEC={current_dec:.2f}°")
        mount.goto_coordinates(target_ra, current_dec, track=False)
        time.sleep(5)

        # Get the current coordinates after RA rotation
        current_ra, current_dec = mount.get_current_coord()
        print(f"Position after RA rotation: RA={current_ra:.2f}°, DEC={current_dec:.2f}°")

        # Rotate 30 degrees on DEC axis
        target_dec = current_dec + 30.0
        if target_dec > 90.0:
            target_dec = 90.0 - (target_dec - 30.0)  # Wrap around the pole

        print(f"Rotating 30 degrees on DEC axis to: RA={current_ra:.2f}°, DEC={target_dec:.2f}°")
        mount.goto_coordinates(current_ra, target_dec, track=False)

        print("Test completed successfully")

    except Exception as e:
        print(f"Error during test: {str(e)}")
    finally:
        # Always ensure proper shutdown
        print("Shutting down mount controller...")
        MountController.shutdown()

if __name__ == "__main__":
    main()
