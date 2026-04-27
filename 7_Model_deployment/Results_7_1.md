# Tulokset ja oma arviointi

[Ajettu notebook](huggingface_model_test_NN.ipynb)

### Arviointi

Tässä tehtävässä ei ollut jokaiselle vaatimukselle määritettyä pistearvoa

1. Asenna NGINX-ympäristö ja testaa esimerkkinettisivua, joka luokittelee kissoja/koiria:

    https://gitlab.dclabra.fi/ai-courses/ai-modern-methods/-/blob/master/examples/tensorflowjs/tensorflowjs_example.zip 

    - Ajoin [esimerkkimallin](tensorflowjs_example) pythonin virtuaaliympäristössä yksinkertaisesti komennolla `python -m http.server 8000`, malli toimii selaimessa kuten kuuluukin

    ![Malli toimii 1](images/esimerkki1.png)

    ![Malli toimii 2](images/esimerkki2.png)

2. Ota tehtävän 2.2 neuroverkko Flowers_SavedModel_NN.zip ja käännä se tensorflow.js muotoon työkalulla https://github.com/tensorflow/tfjs/tree/master/tfjs-converter. 

    - Konversio terminaalissa

    ![Konversio](images/conversion_terminal.png)

    - Lopputulos

    ![Lopputulos](images/conversion_folder.png)

3. Muokkaa esimerkkinettisivua, jotta saat tallentamasi kukka-verkon toimimaan siinä. 

    - Pienten muokkausten jälkeen sivu toimii kukkamallilla ongelmitta

    ![Malli toimii 3](images/esimerkki3.png)

4. Lisää esimerkkinettisivulle myös ennusteen varmuus %-arvona.

    - Mallin varmuus näkyy kuten pitääkin

    ![Malli toimii 4](images/esimerkki4.png)


5. Paketoi muokkaamasi nettisivu ja kukka-verkko tiedostoon kukkaverkko_NN.zip (tai kukkaverkko_NN.tar.gz) ja palauta se tähän.

    [Paketoitu verkko](kukkaverkko_NN.zip)

Vaikka tehtävässä ei ollut määritetty osuuskohtaisia pisteitä, sain mielestäni kaikki osuudet tehtyä oikein.