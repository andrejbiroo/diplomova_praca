import html
from flask import Flask, render_template
import pandas as pd
from bs4 import BeautifulSoup
import requests
import json
import nltk
from nltk.corpus import stopwords
import networkx as nx
from pyvis.network import Network
from collections import Counter

app = Flask(__name__)

@app.route("/")
def home():
    r = requests.get("https://www.ukf.sk/univerzita")
    r.encoding = 'utf-8'
    html_content = r.text

    soup = BeautifulSoup(html_content, 'html.parser')
    divs = soup.find_all('div', {'style': 'text-align: justify;'})
    for div in divs:
        text = div.get_text()
        print(text)

    pracovny_text = 'Jednou z hlavných činností univerzity je výskum. Je zameraný na získavanie pôvodných výsledkov smerujúcich k rozvoju poznania a zahŕňa aj aktivity zamerané na efektívne prepojenie vedeckého bádania so vzdelávacím procesom a na podporu odborného rastu zamestnancov. Vedeckovýskumná činnosť sa uskutočňuje predovšetkým prostredníctvom riešenia výskumných, kultúrno-edukačných a ďalších projektov, pričom nosným výstupom sú vedecké a odborné publikácie. Univerzita každoročne organizuje významné vedecké, umelecké a odborné podujatia, prostredníctvom ktorých sa realizuje prezentácia, propagácia a prenos poznatkov a dosiahnutých výsledkov z riešenia vedeckovýskumných projektov.'

    response = requests.get("http://lindat.mff.cuni.cz/services/udpipe/api/process?tokenizer&tagger&parser&data="+pracovny_text+"&model=slovak-snk-ud-2.5-191206")
    response_text = response.text

    vysledok = json.loads(response_text)
    vysledok_result = vysledok["result"]

    vsetky_riadky_povodne = vysledok_result.split('\n\n')
    vsetky_riadky_povodne_zoznam = []
    for riadok in vsetky_riadky_povodne:
        vsetky_riadky_povodne_zoznam.append(riadok) 

    vsetky_riadky = vysledok_result.split('\n\n')
    vsetky_riadky_zoznam = []
    for riadok in vsetky_riadky:
        vsetky_riadky_zoznam.append(riadok)

    vsetky_riadky[0] = vsetky_riadky[0].replace("# newdoc\n# newpar\n", "")

    zoznam_zoznamov_dvojic = []
    for riadok in vsetky_riadky:
        riadokk = riadok.split('\n')
        df = pd.DataFrame(columns=['riadok'])
        for i in riadokk:
            df = pd.concat([df, pd.DataFrame({'riadok': [i]})], ignore_index=True)
        zoznam_zoznamov_dvojic.append(df)

    zoznam_zoznamov_dvojic[0]['riadok'] = zoznam_zoznamov_dvojic[0]['riadok'].str.replace('\t', ' ', regex=True)

    zoznam_zoznamov_dvojic[0] = pd.concat([zoznam_zoznamov_dvojic[0]['riadok'].str.split(' ', expand=True)], axis=1)

    zoznam_zoznamov_dvojic[0] = zoznam_zoznamov_dvojic[0].drop(zoznam_zoznamov_dvojic[0][zoznam_zoznamov_dvojic[0][0]== '#'].index)

    zoznam_zoznamov_dvojic[0] = zoznam_zoznamov_dvojic[0].dropna(axis=1, how='all')

    zoznam_zoznamov_dvojic[0].columns = ['porad_cislo', 'slovo', 'lema_slova', 'univerzalna_znacka', 'specificka_znacka', 
                                        'feats', 'head', 'deprel','-','-']

    zoznam_zoznamov_dvojic[0] = zoznam_zoznamov_dvojic[0].reset_index(drop=True)

    zoznam_zoznamov_dvojic[0]["dvojica"] = ""

    for i in range(0, len(zoznam_zoznamov_dvojic[0])):
        porad_cislo_index = int(zoznam_zoznamov_dvojic[0]["porad_cislo"].iloc[i]) - 1
        head_index = int(zoznam_zoznamov_dvojic[0]["head"].iloc[i]) - 1
        dvojica = zoznam_zoznamov_dvojic[0]["lema_slova"].iloc[porad_cislo_index]+" "+zoznam_zoznamov_dvojic[0]["lema_slova"].iloc[head_index]
        zoznam_zoznamov_dvojic[0].at[i,'dvojica'] = dvojica 

    zoznam_dvojic_prva_veta = zoznam_zoznamov_dvojic[0]['dvojica'].to_list()

    zoznam_zoznamov_dvojic[1]['riadok'] = zoznam_zoznamov_dvojic[1]['riadok'].str.replace('\t', ' ', regex=True)

    zoznam_zoznamov_dvojic[1] = pd.concat([zoznam_zoznamov_dvojic[1]['riadok'].str.split(' ', expand=True)], axis=1)

    zoznam_zoznamov_dvojic[1] = zoznam_zoznamov_dvojic[1].drop(zoznam_zoznamov_dvojic[1][zoznam_zoznamov_dvojic[1][0]== '#'].index)

    zoznam_zoznamov_dvojic[1] = zoznam_zoznamov_dvojic[1].dropna(axis=1, how='all')

    zoznam_zoznamov_dvojic[1].columns = ['porad_cislo', 'slovo', 'lema_slova', 'univerzalna_znacka', 'specificka_znacka', 
                                        'feats', 'head', 'deprel','-','-']

    zoznam_zoznamov_dvojic[1] = zoznam_zoznamov_dvojic[1].reset_index(drop=True)

    zoznam_zoznamov_dvojic[1]["dvojica"] = ""

    for i in range(0, len(zoznam_zoznamov_dvojic[1])):
        porad_cislo_index = int(zoznam_zoznamov_dvojic[1]["porad_cislo"].iloc[i]) - 1
        head_index = int(zoznam_zoznamov_dvojic[1]["head"].iloc[i]) - 1
        dvojica = zoznam_zoznamov_dvojic[1]["lema_slova"].iloc[porad_cislo_index]+" "+zoznam_zoznamov_dvojic[1]["lema_slova"].iloc[head_index]
        zoznam_zoznamov_dvojic[1].at[i,'dvojica'] = dvojica 

    zoznam_dvojic_druha_veta = zoznam_zoznamov_dvojic[1]['dvojica'].to_list()

    zoznam_zoznamov_dvojic[2]['riadok'] = zoznam_zoznamov_dvojic[2]['riadok'].str.replace('\t', ' ', regex=True)

    zoznam_zoznamov_dvojic[2] = pd.concat([zoznam_zoznamov_dvojic[2]['riadok'].str.split(' ', expand=True)], axis=1)

    zoznam_zoznamov_dvojic[2] = zoznam_zoznamov_dvojic[2].drop(zoznam_zoznamov_dvojic[2][zoznam_zoznamov_dvojic[2][0]== '#'].index)

    zoznam_zoznamov_dvojic[2] = zoznam_zoznamov_dvojic[2].dropna(axis=1, how='all')

    zoznam_zoznamov_dvojic[2].columns = ['porad_cislo', 'slovo', 'lema_slova', 'univerzalna_znacka', 'specificka_znacka', 
                                        'feats', 'head', 'deprel','-','-']

    zoznam_zoznamov_dvojic[2] = zoznam_zoznamov_dvojic[2].reset_index(drop=True)

    zoznam_zoznamov_dvojic[2]["dvojica"] = ""

    for i in range(0, len(zoznam_zoznamov_dvojic[2])):
        porad_cislo_index = int(zoznam_zoznamov_dvojic[2]["porad_cislo"].iloc[i]) - 1
        head_index = int(zoznam_zoznamov_dvojic[2]["head"].iloc[i]) - 1
        dvojica = zoznam_zoznamov_dvojic[2]["lema_slova"].iloc[porad_cislo_index]+" "+zoznam_zoznamov_dvojic[2]["lema_slova"].iloc[head_index]
        zoznam_zoznamov_dvojic[2].at[i,'dvojica'] = dvojica 

    zoznam_dvojic_tretia_veta = zoznam_zoznamov_dvojic[2]['dvojica'].to_list()

    zoznam_zoznamov_dvojic[3]['riadok'] = zoznam_zoznamov_dvojic[3]['riadok'].str.replace('\t', ' ', regex=True)

    zoznam_zoznamov_dvojic[3] = pd.concat([zoznam_zoznamov_dvojic[3]['riadok'].str.split(' ', expand=True)], axis=1)

    zoznam_zoznamov_dvojic[3] = zoznam_zoznamov_dvojic[3].drop(zoznam_zoznamov_dvojic[3][zoznam_zoznamov_dvojic[3][0]== '#'].index)

    zoznam_zoznamov_dvojic[3] = zoznam_zoznamov_dvojic[3].dropna(axis=1, how='all')

    zoznam_zoznamov_dvojic[3].columns = ['porad_cislo', 'slovo', 'lema_slova', 'univerzalna_znacka', 'specificka_znacka', 
                                        'feats', 'head', 'deprel','-','-']

    zoznam_zoznamov_dvojic[3] = zoznam_zoznamov_dvojic[3].reset_index(drop=True)

    zoznam_zoznamov_dvojic[3]["dvojica"] = ""

    for i in range(0, len(zoznam_zoznamov_dvojic[3])):
        porad_cislo_index = int(zoznam_zoznamov_dvojic[3]["porad_cislo"].iloc[i]) - 1
        head_index = int(zoznam_zoznamov_dvojic[3]["head"].iloc[i]) - 1
        dvojica = zoznam_zoznamov_dvojic[3]["lema_slova"].iloc[porad_cislo_index]+" "+zoznam_zoznamov_dvojic[3]["lema_slova"].iloc[head_index]
        zoznam_zoznamov_dvojic[3].at[i,'dvojica'] = dvojica 

    zoznam_zoznamov_dvojic[3] = zoznam_zoznamov_dvojic[3].drop(zoznam_zoznamov_dvojic[3][zoznam_zoznamov_dvojic[3]['specificka_znacka'] == 'Z'].index)

    zoznam_dvojic_stvrta_veta = zoznam_zoznamov_dvojic[3]['dvojica'].to_list()

    celkovy_zoznam_dvojic = zoznam_dvojic_prva_veta + zoznam_dvojic_druha_veta + zoznam_dvojic_tretia_veta + zoznam_dvojic_stvrta_veta

    celkovy_zoznam_dvojic_bez_interpunkcie = [riadok for riadok in celkovy_zoznam_dvojic if "." not in riadok and "," not in riadok and "!" not in riadok and "?" not in riadok]

    dvojice = [(t.split()[0], ' '.join(t.split()[1:])) for t in celkovy_zoznam_dvojic_bez_interpunkcie]

    stop_words = stopwords.words('slovak')
    dvojice = [list(set(sublist).difference(stop_words)) for sublist in dvojice]
    dvojice = [[word for word in sublist if word.lower() not in stop_words] for sublist in dvojice]
    dvojice = [tup for tup in dvojice if len(tup) > 1]

    G = nx.Graph()
    G.add_edges_from(dvojice)

    pocet_dvojic = Counter(tuple(edge) for edge in dvojice)

    nt = Network('1000px', '1000px', notebook=True)

    for edge in dvojice:
        source, target = edge[:2]
        nt.add_node(source, label=source)
        nt.add_node(target, label=target)
        nt.add_edge(source, target)  

    nt.show_buttons(filter_=['physics'])

    for node in nt.nodes:
        pocet_dvojic = sum(1 for edge in dvojice if node["id"] in edge[:2])
        node["label"] += "\n(počet dvojíc: {})".format(pocet_dvojic)

    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)