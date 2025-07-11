#!/bin/bash

if [ ! -x "$0" ]; then
    chmod +x "$0"
    bash "$0"
    exit 0
fi

if [ "$1" == "--IUseArchBtw" ]; then
    echo "Installing on Arch Linux..."
    echo "→ Installing base-devel, Python 3.10, and virtualenv"
    sudo pacman -Syu --needed --noconfirm \
        base-devel \
        python \
        python-virtualenv

    echo
    echo "✅ Installed Python3.10 on Arch."
else
    echo "Installing on Debian/Ubuntu..."
    echo "→ Updating package lists"
    sudo add-apt-repository ppa:deadsnakes/ppa
    sudo apt-get update

    echo "→ Installing build-essential, Python 3, venv, and pip"
    sudo apt-get install -y \
        build-essential \
        python3.10 \
        python3.10-venv \
        python3.10-dev

    # (Optional) If you still need the old 'virtualenv' command:
    # sudo pip3 install virtualenv

    echo
    echo "✅ Installed Python3.10 on Debian/Ubuntu."
fi


if ! command -v python3.10 &>/dev/null; then
    echo "Python3 installation failed. Exiting..."
    exit 1
fi

VENV_NAME="qnetfires"

if [ -d "$VENV_NAME" ]; then
    echo "Virtual environment already exists. Skipping creation..."
else
    echo "Creating virtual environment named: '$VENV_NAME'..."
    python3.10 -m venv "$VENV_NAME"
    echo "Virtual environment '$VENV_NAME' created."
fi

if [ -n "$VIRTUAL_ENV" ] && [ "$VIRTUAL_ENV" == "$(pwd)/$VENV_NAME" ]; then
    echo "Already inside the virtual environment '$VENV_NAME'"
else
    echo "Activating virtual environment: '$VENV_NAME'..."
    source "$VENV_NAME/bin/activate"
fi

echo "Upgrading pip..."
pip install --upgrade pip setuptools
echo "Upgraded pip."

if [ -f "requirements.txt" ]; then
    echo "Installing python dependencies..."
    pip install -r requirements.txt
    echo "Finished installing dependencies."
else
    echo "Warning: requirements.txt not found. Exiting installation..."
    echo "Please consider a re-pull of the repository and try installing again. Simply run:
    git pull git@github.com:mengsig/QNetFires.git"
    exit 1
fi

echo "[DomiRank]: Installing DomiRank library and dependencies..."
git clone git@github.com:mengsig/DomiRank.git
pip install -e DomiRank/.
echo "[DomiRank]: Finished installing DomiRank library and dependencies!"

echo "[Pyregenics]: Installing Pyregenics library and dependencies..."
git clone git@github.com:pyregence/pyretechnics.git
cd pyretechnics
python setup.py install
cd ..
echo "[Pyregenics]: Finished installing pyregenics library and dependencies!"

echo "Setup Complete! Virtual environment '$VENV_NAME' has now been successfully created."

FILENAME_ENV="$(pwd)/env.sh"
if [ ! -f "$FILENAME_ENV" ]; then
    echo "Cannot find the '$FILENAME_ENV' file. Please re-pull via:
    git pull git@github.com:mengsig/QNetFires.git"
    echo "Stopping installation..."
    exit 1
fi

if [ ! -x "$FILENAME_ENV" ]; then
    chmod +x "$FILENAME_ENV"
    echo "Created executable name '$FILENAME_ENV'"
fi

echo "Downloading necessary .tif files..."
python src/scripts/Download.py
echo "Finished downloading .tif files!"

echo "Creating DomiRank Fuel-Breaks necessary for pre-training..."
python src/scripts/CreateAdjacency.py
echo "Finished creating DomiRank Fuel-Breaks!"

echo "To activate the virtual environment at any time, run the command:
    source env.sh"
