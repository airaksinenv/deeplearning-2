TEHTÄVÄ: 

Valitse https://huggingface.co/models -kirjastosta kaksi (2) erilaista neuroverkkoa ja testaa niiden toimintaa yhdessä notebookissa. Minä testasin esimerkiksi kahta tällaista mallia:

- Toxicity analysis with RobertaForSequenceClassification (osasi ennustaa oikein lyhyitä lauseita kuten "I hate you!" tai "I love you!")

- Segmentation with MobileViTForSemanticSegmentation (osasi segmentoida kissat oikein kuvasta)

Palauta:

    huggingface_model_test_NN.ipynb


Arviointi:

1. Notebookin ajo toimii jupyterhubissa /test -kansiossa ilman ongelmia ja ajoaika on alle 10min  (läpäisyehto)
2. Notebookissa ladataan ja ajetaan kahta eri huggingface mallia (1p / malli)
3. Notebookissa testataan malleja omalla datalla ja tulostetaan tulokset  (1p / malli)
4. Notebookissa pohditaan saatuja tuloksia  (1p / malli)