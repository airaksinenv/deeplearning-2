TEHTÄVÄ: Lisää MLflow Tracking tehtävän 4.1 train_flower_model_NN.py scriptiin ja aja koulutus niin, että experiment näkyy projektin koulutus experimenteissä. Käytä seuraavia parametreja: 

mlflow.set_tracking_uri: "/scratch/project_2015254/mlruns"
mlflow.set_experiment: "flowers"
run_name: NN+jobid

Palauta:

    Experiment näkyy MLflow käyttöliittymältä. (Tähän ei siis palauteta koodia ollenkaan.) Kuvakaappaukset riittävät.

Arviointi:

1. Experiment näkyy MLflow käyttöliittymältä. (2p)
2. Experimentissä näkyy loss, val_loss, accuracy ja val_accuracy käyrät (1p)
3. Experimentistä löytyy model Artifacts -näkymästä. (1p)
4. Tarkkuus on > 70% (1p)
