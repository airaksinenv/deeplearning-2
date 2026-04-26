TEHTÄVÄ: Muokkaa Palautuslaatikko 2.1: TF_exercise_flowers_with_data_augmentation.ipynb notebookia niin, että se kouluttaa neuroverkkoa niin monella GPU:lla kuin ympäristössä on, eli skaalautuu automaattisesti.

Muokkaukset:

1. Konfiguroi yksi fyysinen GPU neljäksi (4) loogiseksi GPU:ksi käyttäen funktiota tf.config.set_logical_device_configuration().
2. Päivitä GLOBAL_BATCH_SIZE -koko vastaamaan käytössä olevia GPU prosessoreja.
3. Ota käyttöön keras.utils.image_dataset_from_directory() datasetit sekä training että validointi datoille.
4. Aseta distribution strategy ja muodosta & käännä neuroverkko käyttäen tätä strategiaa.
5. Kouluta neuroverkkoa N epochia [esim. 5-30].


HUOM. Notebookissa ei tarvitse olla mukana yhden GPU:n koulutusta, mutta koodi kannattaa tehdä niin, että se tukee N määrää GPU:ta.
Palauta:

`Notebook TF_exercise_flowers_multiGPU_NN.ipynb`


Arviointi:

- Notebookin ajo toimii jupyterhubissa /test -kansiossa ilman ongelmia ja ajoaika on alle 15 min (2p)
- Neuroverkon tarkkuus validointidatalla on > 70% (1p)
- Notebookin viimeinen solu tulostaa arvot: # of replicas, training_accuracy, validation_accuracy, training_time. (2p)

Huom. Älä käytä tehtävässä pohjana valmista neuroverkkoa (= transfer learning), vaan määritä oma neuroverkko, jonka koulutat alusta asti. => -1p
