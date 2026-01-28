"""
Quick Start Script
==================
One-click setup and run script for the Solar Panel Efficiency Prediction project.

Usage:
    python run.py [command]

Commands:
    setup    - Generate data and train model
    train    - Train the model only
    app      - Run the web application
    all      - Setup and run application
"""

import subprocess
import sys
import os


def run_command(command: str, description: str) -> bool:
    """Run a shell command and return success status."""
    print(f"\n{'='*60}")
    print(f">>> {description}")
    print(f"{'='*60}\n")
    
    result = subprocess.run(command, shell=True)
    return result.returncode == 0


def install_dependencies():
    """Install required dependencies."""
    print("Installing dependencies...")
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )


def generate_data():
    """Generate training data."""
    return run_command(
        f"{sys.executable} src/data_generator.py",
        "Generating synthetic dataset"
    )


def train_model():
    """Train the deep learning model."""
    return run_command(
        f"{sys.executable} src/train.py --model-type deep --epochs 100",
        "Training deep learning model"
    )


def run_app():
    """Run the Streamlit web application."""
    print("\n" + "="*60)
    print(">>> Starting Web Application")
    print("="*60)
    print("\nOpen your browser and navigate to: http://localhost:8501")
    print("Press Ctrl+C to stop the server.\n")
    
    subprocess.run(f"{sys.executable} -m streamlit run app.py", shell=True)


def setup():
    """Complete setup: generate data and train model."""
    print("\n" + "="*60)
    print("SOLAR PANEL EFFICIENCY PREDICTION")
    print("Complete Setup")
    print("="*60)
    
    # Generate data
    if not generate_data():
        print("Failed to generate data!")
        return False
    
    # Train model
    if not train_model():
        print("Failed to train model!")
        return False
    
    print("\n" + "="*60)
    print("SETUP COMPLETE!")
    print("="*60)
    print("\nRun 'python run.py app' to start the web application.")
    return True


def main():
    """Main entry point."""
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable commands: setup, train, app, all")
        return
    
    command = sys.argv[1].lower()
    
    if command == 'setup':
        setup()
    elif command == 'train':
        train_model()
    elif command == 'app':
        run_app()
    elif command == 'all':
        if setup():
            run_app()
    elif command == 'install':
        install_dependencies()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: setup, train, app, all, install")


if __name__ == "__main__":
    main()
