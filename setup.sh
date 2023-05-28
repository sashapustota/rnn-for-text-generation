# Create a virtual environment
python3 -m venv rnn-for-text-generation-venv

# Activate the virtual environment
source ./rnn-for-text-generation-venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

# To remove the virtual environment run the following command in the terminal
#rm -rf rnn-for-text-generation-venv