TEHTÄVÄ: 

Phase 1: Aja tehtävän 3.2 train_flower_model_NN.py -scripti CSC:n PUHTI supertietokoneella.

Ohjeet - osa 1 

Vinkki: Testaa ensin scriptin ajoa muutamalla EPOCH:illa ilman GPU-varausta. Kun tämä toimii, lisää EPOCH:ien määrää ja ota 1xGPU käyttöön. 

Phase 2: Muokkaa scriptejä seuraavasti:

- Lisää batch scriptin loppuun test_flower_model_NN.py scriptin ajo (muokkaa python scriptiä niin, että sen printit menevät test_log.out -tiedostoon.) 
- Ota käyttöön laskentasolmun paikallinen nopea levy kopioimalla koulutus/validointi/testaus -data Altaasta sille ennen python scriptien ajoa
- Ota käyttöön laskentasolmun kaikki 4 GPU:ta

Ohjeet - osa 2 

Palauta:

seuraavat vaiheen 2 tiedostot yhdessä <PUHTI_Etunimi_Sukunimi.zip> -paketissa:

        Flower_script.sh
        training_script.py
        train_log.out
        test_script.py
        test_log.out
        slurm-<task-id>.out

Varmista, että koulutettu neuroverkko löytyy polusta: `/projappl/<project_id>/models/flower_model_NN.keras`


Arviointi:

1. Scriptien ajo toimii paketin kopioinnin ja purkamisen jälkeen PUHTI supertietokoneen gputest partitiossa ilman ongelmia ja ajoaika on alle 15 min (2p)
2. /projappl/<project_id>/models/ -kansioon tallentuu flower_model_NN.keras malli (1p)
3. Neuroverkon tarkkuus validointidatalla on > 60% (1p)
4. Koulutus/test -data luetaan Altaasta paikalliselle levylle (2p)
5. Koulutus tapahtuu neljällä GPU:lla (2p)
6. Logien lopusta löytyy seuraavat tulostukset: # of replicas (=GPU count), training_accuracy, validation_accuracy, run_time. (1p)
