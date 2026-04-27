# Tulokset ja oma arviointi

[Ajettu notebook](tokenizers_NN.ipynb)

### Arviointi

Tässä tehtävässä ei ollut jokaiselle vaatimukselle määritettyä pistearvoa

1. Notebook ja siinä olevat 2 tokenizeria toimii. (3p)

    - [Notebook](tokenizers_NN.ipynb) ja siinä olevat tokenizerit toimivat. 3/3 pistettä.

2. Notebookissa on pohdittu tokeniserien välisiä eroja (1p)

    - Pohdinta alla. 1 piste.

    ## Pohdinta: BPE vs WordPiece

    BPE ja WordPiece ovat molemmat alisanoihin perustuvia tokenisointimenetelmiä, mutta niiden merkintätapa ja tokenien muodostus eroavat hieman.

    Tässä testissä **BPE** pilkkoo sanoja usein lyhyisiin merkkijonoihin ja pyrkii muodostamaan yleisiä merkkipareja tokeniksi. Koska aineisto on pieni, BPE ei opi kovin laajaa sanastoa, joten osa pidemmistä sanoista pilkkoutuu useiksi pieniksi osiksi.

    **WordPiece** käyttää tässä `##`-merkintää jatko-osille. Tämä tekee näkyväksi sen, mitkä tokenit ovat sanan alussa ja mitkä ovat sanan jatkoa. Esimerkiksi pitkä sana voi näkyä muodossa `san`, `##an`, `##loppu`. Tämä on selkeä ero BPE-tokenisointiin, jossa jatko-osia ei merkitä samalla tavalla.

    Käytännössä molemmat menetelmät auttavat käsittelemään sanoja, joita ei löydy kokonaisina sanastosta. Tämän takia ne ovat hyödyllisiä neuroverkoissa: mallin ei tarvitse tuntea jokaista sanaa kokonaisena, vaan se voi käsitellä sanan osina.

---

3. Notebookissa on pohdittu tokeniserien eroja eri kielien välillä (1p)

    - Pohdinta alla. 1 piste.

    ## Pohdinta: kielten väliset erot

    Englanninkielinen teksti tokenisoituu yleensä hieman yksinkertaisemmin, koska sanat ovat tässä esimerkissä melko lyhyitä ja monia yleisiä sanoja esiintyy sellaisenaan. Esimerkiksi sanat kuten `alice`, `sister` ja `book` voivat säilyä kokonaisina tai lähes kokonaisina tokenina.

    Suomenkielisessä tekstissä sanat ovat pidempiä ja sisältävät enemmän taivutusta. Esimerkiksi `istuessaan`, `sisarensa`, `keskusteluja` ja `kurkistanut` sisältävät vartalon lisäksi päätteitä tai johdoksia. Tällaiset sanat pilkkoutuvat helpommin useiksi alisanoiksi.

    Tämä havainnollistaa, miksi suomen kaltaisissa kielissä tokenimäärä voi kasvaa verrattuna englantiin. Morfologisesti rikkaassa kielessä sama perusmerkitys voi esiintyä monessa taivutusmuodossa, jolloin alisanatokenisointi on erityisen tärkeää.

Oma arvioni tehtävästä on 5/5 pistettä.