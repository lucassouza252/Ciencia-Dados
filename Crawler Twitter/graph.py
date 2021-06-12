import numpy as np
import matplotlib.pyplot as plt


def plotgraph(common, lista_cinco):
    x_bar = []
    y_bar = []
    fatias = []
    legendas = []
    for i in range(len(common)):
        x_bar.append(common[i][0])
        y_bar.append(common[i][1])

    for i in range(len(lista_cinco)):
        legendas.append(common[i][0])
        fatias.append(common[i][1])

    fig = plt.figure()

    axes1 = fig.add_axes([0.1, 0.1, 0.8, 0.8]) # axes da figura principal
    axes2 = fig.add_axes([0.5, 0.6, 0.4, 0.3]) # axes da figura secundaria
    
    axes1.barh(x_bar, y_bar)
    axes1.set_xlabel("Frequência")
    axes1.set_ylabel("Termos")
    axes1.set_title("Frequência de Termos")

    axes2.pie(fatias, labels=legendas, startangle=90, shadow=True)
    axes2.set_title("5 Palavras Mais faladas")

    plt.savefig("listadepalavras.png")

def cinco_maior(common):
    maiores = []
    for i in range(len(common)):
        maior = common[i][1]
        for j in range(i, len(common)):
            if maior in maiores:
                continue
            if maior < common[j][1]:
                maior = common[j][1]
        maiores.append(maior)
    return (maiores[0:5])

def lista_cinco_maior(common, maiores):
    lista_cinco = []
    for i in range(len(common)):
        if common[i][1] in maiores:
            lista_cinco.append(common[i])
    return lista_cinco