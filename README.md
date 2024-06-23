python -m venv openvino_env
openvino_env\Scripts\activate
python -m pip install --upgrade pip
python -m pip install openvino

#if you dont have gpu in your device then in requirements.txt
    #remove following packages from requirements.txt
        torch==2.3.1+cu121
        torchaudio==2.3.1+cu121
        torchvision==0.18.1+cu121

    #and run the given command on terminal
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt