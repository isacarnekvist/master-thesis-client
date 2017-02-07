sudo apt-get install python3-pip python3-scipy -y
sudo pip3 install keras
sudo pip3 install theano
sudo pip3 install pyuarm
sudo pip3 install logcolor
sudo pip3 install requests
mkdir -p $HOME/.keras
echo '{"image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' > $HOME/.keras/keras.json
git clone git@github.com:isacarnekvist/naf.git
