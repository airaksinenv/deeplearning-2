TEHTÄVÄ: Tee kaksi (2) python scriptiä:

A. Muokkaa edellisen tehtävän notebook scriptiksi train_flower_model_NN.py, joka
sisältää funktion: train_flower_model("pathToTrainDataset", "pathToValidationDataset")

- supports multi GPU training
- uses also validation data to see train-validation difference vs epochs
- saves training logs to file "train_log.out"
    -   Including prints: # replicas / global batch size / training data count / model summary / results 
- saves model as Keras model to file "./models/flower_model_NN.keras"
- No return value

B. Tee lisäksi scripti test_flower_model_NN.py, joka
sisältää funktion: test_flower_model("pathToTestDataset")

- loads model (.keras) 
- saves test logs to file "test_log.out"
    -   Including prints: UTC time / run time / tested images / correct % / test loss per image 
- return values: tested images / correct % / test loss per image

Palauta: 

1. Script train_flower_model_NN.py 
2. Script test_flower_model_NN.py


Arviointi:

- Scriptien peräkkäinen ajo toimii jupyterhubissa ilman ongelmia ja yhteinen ajoaika on alle 15min (2p)
- Neuroverkon tarkkuus test idatalla on > 60% (1p)
- Kaikki vaaditut tiedostot, printit ja paluu arvot löytyy. (2p)

