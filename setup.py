#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script di Setup per il Progetto "Hello Coral"
Autore: Craicek

Questo script esegue tutte le operazioni di preparazione
necessarie per eseguire 'main.py'.

FASE 1: VERIFICA AMBIENTE VIRTUALE
        Controlla che lo script sia eseguito all'interno di un
        ambiente virtuale (venv) per evitare di installare
        pacchetti a livello globale.

FASE 2: INSTALLAZIONE DIPENDENZE
        Esegue 'pip install -r requirements.txt' per installare
        tutte le librerie Python necessarie (OpenCV, PyCoral, ecc.)

FASE 3: DOWNLOAD FILE MODELLO E ETICHETTE
        Crea la cartella 'modelli' e scarica il modello
        '.tflite' compilato per la Edge TPU e il file
        '.txt' con le etichette delle classi.
"""

import os
import urllib.request
import subprocess
import sys

# --- COSTANTI ---
NOME_CARTELLA_MODELLI = "modelli"
NOME_FILE_REQS = "requirements.txt"

# URL ufficiali di Google per il modello e le etichette
URL_MODELLO = "https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
NOME_FILE_MODELLO = "mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"

URL_ETICHETTE = "https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt"
NOME_FILE_ETICHETTE = "coco_labels.txt"


def check_venv():
    """
    Verifica se lo script è in esecuzione in un ambiente virtuale.
    Restituisce True se in un venv, False altrimenti.
    """
    # 'sys.prefix' è la directory dell'ambiente Python corrente
    # 'sys.base_prefix' è la directory dell'installazione Python di sistema
    # Se sono diversi, siamo in un venv.
    return (hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))


def installa_dipendenze():
    """
    Esegue 'pip install -r requirements.txt' usando subprocess.
    Restituisce True se l'installazione ha successo, False altrimenti.
    """
    if not os.path.exists(NOME_FILE_REQS):
        print(f"✗ ERRORE: File '{NOME_FILE_REQS}' non trovato.")
        return False

    print(f"Installazione delle dipendenze da '{NOME_FILE_REQS}'...")
    try:
        # Usiamo 'sys.executable' per essere sicuri di usare
        # il 'python' e 'pip' dell'ambiente virtuale corrente.
        comando_pip = [sys.executable, "-m", "pip", "install", "-r", NOME_FILE_REQS]

        # 'check=True' solleverà un'eccezione se pip fallisce
        result = subprocess.run(comando_pip, check=True, capture_output=True, text=True, encoding='utf-8')

        print(result.stdout)  # Mostra l'output di pip
        print("✓ Dipendenze installate con successo.")
        return True

    except subprocess.CalledProcessError as e:
        # Errore catturato da pip (es. pacchetto non trovato)
        print("✗ ERRORE durante l'installazione delle dipendenze con pip:")
        print(e.stderr)
        return False
    except FileNotFoundError:
        print("✗ ERRORE: Impossibile trovare 'pip'.")
        return False


def scarica_file_modello():
    """
    Crea la cartella 'modelli' e scarica il modello e le etichette.
    Restituisce True se tutti i file sono pronti, False altrimenti.
    """
    # 1. Crea la cartella 'modelli'
    if not os.path.exists(NOME_CARTELLA_MODELLI):
        print(f"Creazione della cartella: {NOME_CARTELLA_MODELLI}")
        os.makedirs(NOME_CARTELLA_MODELLI)
    else:
        print(f"✓ Cartella '{NOME_CARTELLA_MODELLI}' già esistente.")

    # Percorsi completi ai file
    path_modello = os.path.join(NOME_CARTELLA_MODELLI, NOME_FILE_MODELLO)
    path_etichette = os.path.join(NOME_CARTELLA_MODELLI, NOME_FILE_ETICHETTE)

    # Flag per tracciare il successo
    successo = True

    # 2. Scarica il modello TFLite
    if not os.path.exists(path_modello):
        print(f"Download del modello MobileNet SSD v2...")
        try:
            urllib.request.urlretrieve(URL_MODELLO, path_modello)
            print("✓ Download modello completato.")
        except Exception as e:
            print(f"✗ Errore durante il download del modello: {e}")
            successo = False
    else:
        print("✓ File modello già presente.")

    # 3. Scarica il file delle etichette
    if not os.path.exists(path_etichette):
        print(f"Download del file etichette COCO...")
        try:
            urllib.request.urlretrieve(URL_ETICHETTE, path_etichette)
            print("✓ Download etichette completato.")
        except Exception as e:
            print(f"✗ Errore during il download delle etichette: {e}")
            successo = False
    else:
        print("✓ File etichette già presente.")

    return successo


def main():
    """
    Flusso principale dello script di setup.
    """
    print("Avvio dello script di setup per il progetto 'Hello Coral'...")
    print("=" * 60)

    # FASE 1: VERIFICA AMBIENTE VIRTUALE
    print("1. Verifica dell'ambiente virtuale...")
    if not check_venv():
        print("✗ ERRORE: Esegui questo script all'interno di un ambiente virtuale (venv).")
        print("Crea un venv ('python -m venv venv') e attivalo prima di continuare.")
        print("=" * 60)
        sys.exit(1)  # Esce con codice di errore
    print("✓ Controllo ambiente virtuale superato.")
    print()

    # FASE 2: INSTALLAZIONE DIPENDENZE
    print("2. Installazione delle dipendenze Python...")
    if not installa_dipendenze():
        print("Esecuzione interrotta a causa di un errore nell'installazione.")
        print("=" * 60)
        sys.exit(1)
    print()

    # FASE 3: DOWNLOAD FILE MODELLO E ETICHETTE
    print("3. Download dei file del modello AI...")
    if not scarica_file_modello():
        print("Esecuzione interrotta a causa di un errore nel download.")
        print("=" * 60)
        sys.exit(1)
    print()

    # Conclusione
    print("=" * 60)
    print("Setup completato con successo!")
    print("Tutte le dipendenze sono installate e i modelli sono scaricati.")
    print("Ora puoi eseguire 'python main.py' per avviare il rilevamento.")
    print("=" * 60)


if __name__ == "__main__":
    main()