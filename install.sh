sudo pip3 install keras
sudo pip3 install theano
mkdir -p $HOME/.keras
echo '{"image_dim_ordering": "tf", "epsilon": 1e-07, "floatx": "float32", "backend": "theano"}' > $HOME/.keras/keras.json
