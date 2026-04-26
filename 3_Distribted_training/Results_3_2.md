# Tulokset ja oma arviointi

[Train script](train_flower_model_NN.py)

[Train log](train_log.out)

[Test script](test_flower_model_NN.py)

[Test log](test_log.out)

[Koulutettu malli](https://github.com/airaksinenv/deeplearning-2/blob/main/3_Distribted_training/models/flower_model_NN.keras)

### Arviointi

1. Scriptien peräkkäinen ajo toimii jupyterhubissa ilman ongelmia ja yhteinen ajoaika on alle 15min (2p)

    - Scriptien ajo toimii ongelmitta ja kesto oli noin ~5min, testi scripti oli hyvin nopea ja melkein kaikki aika meni train scriptissä. 2/2 pistettä.

    ![terminal](images/terminal.png)

2. Neuroverkon tarkkuus testi datalla on > 60% (1p)

    - Tarkkuus pysyi ~70% missä se oli edellisessäkin tehtävässä, ylittää siis vaaditun 60%. 1/1 piste.

    ![test log](images/test_log.png)



3. Kaikki vaaditut tiedostot, printit ja paluu arvot löytyy. (2p)

    -  replicas / global batch size / training data count

    ![train log 1](images/train_log1.png)

    -  model summary

    ![train log 2](images/train_log2.png)

    -  results

    ![train log 3](images/train_log3.png)

    -  UTC time / run time / tested images / correct % / test loss per image 

    ![test log](images/test_log.png)

    - Näistä 2/2 pistettä


Oma arvioini tehtävästä on 5/5 pistettä 