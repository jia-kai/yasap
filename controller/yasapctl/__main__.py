"""Entry point for the YASAP backend application."""

from . import create_app

def main():
    """Run the application."""
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)

if __name__ == '__main__':
    main() 