from .mount import MountController
from .logging import LogManager

from flask import Flask, jsonify, redirect, render_template, request, url_for
import pytz
import yaml

from datetime import datetime
from functools import wraps
from pathlib import Path

# Default configuration
DEFAULT_CONFIG = {
    'latitude': 43.6532,  # Toronto coordinates
    'longitude': -79.3832,
    'elevation': 76,  # Toronto elevation in meters
    #'mount_port': 'tcp://10.10.100.254:8899',
    'mount_port': '/dev/ttyUSB0'
}

# Initialize logger
logger = LogManager.get_instance()

def get_config_path():
    """Get the path to the configuration file."""
    base_dir = Path(__file__).parent
    return base_dir / 'config' / 'config.yaml'

def load_config():
    """Load configuration from file or return defaults."""
    config_path = get_config_path()
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return DEFAULT_CONFIG

def save_config(config):
    """Save configuration to file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

def handle_errors(f):
    """Decorator to handle errors in route handlers."""
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {f.__name__}: {str(e)}")
            return jsonify({'success': False, 'error': str(e)}), 500
    return wrapper

def create_app(test_config=None):
    """Create and configure the Flask application."""
    app = Flask(__name__,
                static_folder=Path(__file__).parent / 'static',
                template_folder=Path(__file__).parent / 'templates')

    if test_config is None:
        app.config.from_mapping(
            SECRET_KEY='dev',
        )
    else:
        app.config.from_mapping(test_config)

    @app.route('/')
    def index():
        # If mount is already set up, redirect to control room
        if MountController.inst() is not None:
            return redirect(url_for('control_room'))

        config = load_config()
        return render_template('index.html', config=config)

    @app.route('/control-room')
    def control_room():
        if MountController.inst() is None:
            return redirect(url_for('index'))
        return render_template('control_room.html')

    @app.route('/api/get_time')
    @handle_errors
    def get_time():
        tz = pytz.timezone('America/Toronto')
        current_time = datetime.now(tz).strftime('%Y-%m-%d %H:%M:%S %Z')
        return jsonify({'success': True, 'time': current_time})

    @app.route('/api/update_gps', methods=['POST'])
    @handle_errors
    def update_gps():
        # TODO: Implement GPS module reading
        logger.info("GPS update requested")
        raise NotImplementedError("GPS module reading not implemented yet")

    @app.route('/api/setup_mount', methods=['POST'])
    @handle_errors
    def setup_mount():
        data = request.get_json()
        config = {
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude']),
            'elevation': float(data['elevation']),
            'mount_port': data['mount_port']
        }

        logger.info(f"Setting up mount with config: {config}")
        MountController.setup(**config)

        # save configuration after successful setup
        save_config(config)
        logger.info("Mount setup completed successfully")

        return jsonify({'success': True})

    @app.route('/api/get_logs')
    @handle_errors
    def get_logs():
        """Get the current log buffer contents."""
        logs = logger.get_log_buffer()
        return jsonify({'success': True, 'logs': logs})

    return app
