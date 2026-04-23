TEHTÄVÄ: Vertaile Pytorch ja TensorFlow kirjastoja opettamalla samanlainen kissa/koira neuroverkko molemmilla. 

Pytorch: https://gitlab.dclabra.fi/ai-courses/ai-deep-learning-2/-/blob/main/examples/pytorch/PyTorch_Cat_Dog_reference_train.ipynb

Tensorflow: https://gitlab.dclabra.fi/ai-courses/ai-deep-learning-2/-/blob/main/examples/keras/TF_Keras_%233_dogs_vs_cats_with_augmentation.ipynb


Muokkaa molemmat ensin seuraavien parametrien mukaisiksi:

- Kuvakoko: 224x224px (3 väriä) 
- Kuvan esikäsittely parametrit: Random flip 50% / random rotation +/- 30as / ei normalisointia 
- minibatch koko: 32
- neuroverkon parametrit:  

conv_blocks (3pcs) : 32 / 64 / 128 channels

A. Convolution: kernel size: 3x3 / stride=1 / no padding

B. aktivation: ReLu

C. Max pooling 

- learning rate: 0.001
- koulutus epochien lkm.: 6
- ajoympäristö: GPU
Vertaile ainakin:

- koulutukseen kulunutta aikaa
- training loss/accuracy
- validation loss/accuracy
- Vaadittua koodin määrää


Palauta vertailu PDF-tiedostona.

Huom. Neuroverkon tarkkuus yllä olevilla parametreilla pitäisi olla ~80%.