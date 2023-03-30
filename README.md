Project for the course Introduction to Deep Learning 2023. Group name O&A

## Loading data

Expects a data directory at root level containing annotations/ and images/

## Running the program

Navigate to src/ directory and run main.py

Otherwise relative paths get messed up and I can't be bothered to make it better

## TODO

* f1-metriikoita

https://learnopencv.com/multi-label-image-classification-with-pytorch-image-tagging/
ja kurssin slackissa on kans jotain

Vois ainaki laskee että meiän f1 on parempi ku dummy :D

* validaatio järkeväks

epochit ylös et se earlystopper tekee jotain.

printtaa jotain et saadaan joku estimate miten hyvä meiän malli on? 

* hyperparametrei ja mallia yms voi parantaa jos jaksaa

Lopuks?

* Tallentaa parhaan mallin että voi sit helposti ajaa testi datan siitä läpi

Lopullinen malli???
* Menee vaan tolla mikä tulee ku laitetaan nykyisellä setillä läpi isoilla epocheilla + stopperilla
* Voi laittaa sitä validaatio splitin määrää pienemmäks että enemmän treenidataa?
* Ajaa semi isol epochi määrällä ja koko datasetillä. ei validaatiota täs vaihees. epochi määrä vaikka se mihin stoppasi ku oli train-validation setit