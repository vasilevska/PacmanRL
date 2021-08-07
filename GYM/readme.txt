 python -m pip install --upgrade pip
 sudo apt-get update
 pip3 install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
 pip install gym
 pip install matplotlib
 pip install gym[atari]
 sudo apt install libtinfo5 #JOK
 sudo apt install libopencv-dev

 wget cutl http://www.atarimania.com/roms/Roms.rar
 sudo apt-get install unrar
 mkdir rars
 unrar x Roms.rar
 python -m atari_py.import_roms rars