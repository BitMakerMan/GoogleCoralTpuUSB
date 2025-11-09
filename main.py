#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
PROGETTO ACCADEMICO "HELLO CORAL" (v1.0)
Autore: Craicek

OBIETTIVO:
Questo script dimostra l'inferenza di Object Detection in tempo reale
utilizzando una webcam e un acceleratore Google Coral TPU.
È progettato per scopi didattici, con un codice pulito e
una documentazione integrata che spiega ogni passaggio.

---
IL CONCETTO CHIAVE: RISOLUZIONE E ASPECT RATIO
Questo progetto risolve il problema più comune nell'IA "Edge":

1. IL CONFLITTO:
   - La nostra WEBCAM ha una risoluzione nativa (es. 1280x720, 16:9).
   - Il MODELLO AI (MobileNet SSD v2) accetta solo un input
     fisso e quadrato (300x300, 1:1).

2. LA SOLUZIONE (Metodo "Squash" o "Schiacciamento"):
   Per massimizzare la velocità e mantenere la semplicità,
   adottiamo questo approccio:

   a. PRE-PROCESSING (Schiacciamento):
      Leggiamo il frame nativo (1280x720) e lo "schiacciamo"
      brutalmente a 300x300 (cv2.resize). L'immagine
      risulta distorta, ma è un'operazione velocissima.

   b. INFERENZA:
      Inviamo l'immagine distorta 300x300 alla TPU, che
      identifica gli oggetti (es. 'person') e restituisce
      le coordinate dei box (Bounding Box) relative
      all'immagine 300x300.

   c. POST-PROCESSING (Correzione della Distorsione):
      Questa è la parte più importante. Dobbiamo mappare
      i box rilevati sull'immagine 300x300 distorta
      sul nostro frame originale 1280x720 non distorto.

      Per farlo, calcoliamo due fattori di scala DIVERSI:
      - scale_x = width_nativo / width_modello (es. 1280 / 300 = 4.26)
      - scale_y = height_nativo / height_modello (es. 720 / 300 = 2.4)

      Usiamo poi la funzione 'bbox.scale(scale_x, scale_y)'
      fornita dalla libreria pycoral. Questa funzione "annulla"
      la distorsione per il box, mappandolo perfettamente
      alle coordinate del frame originale.

3. IL RISULTATO:
   Un'inferenza fluida e in tempo reale (alti FPS) con
   bounding box accurati, indipendentemente dalla risoluzione
   scelta per la webcam.
---

STRUTTURA DEL FILE:
- Sezione 1: Importazioni
- Sezione 2: Costanti Globali
- Sezione 3: Funzioni Helper (Logica di supporto)
- Sezione 4: Blocco Principale (main) (Flusso del programma)
"""

# --- SEZIONE 1: IMPORTAZIONI ---
import cv2  # OpenCV: Per la cattura video, il ridimensionamento e il disegno
import numpy as np  # NumPy: Per la manipolazione di array (usato da OpenCV)
import os  # Per controllare l'esistenza dei file (modello, etichette)
import sys  # Per gestire l'uscita pulita dal menu
import time  # Per calcolare gli FPS (Frame Per Second)

# Importazioni specifiche di PyCoral
from pycoral.utils.edgetpu import make_interpreter  # Per caricare il modello sulla TPU
from pycoral.adapters import common  # Per preparare l'immagine di input
from pycoral.adapters import detect  # Per interpretare i risultati del rilevamento

# --- SEZIONE 2: DEFINIZIONE COSTANTI ---

# Percorsi ai file scaricati da setup.py
PATH_MODELLO = "modelli/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite"
PATH_ETICHETTE = "modelli/coco_labels.txt"

# Soglia di confidenza (0.0 - 1.0)
# Spiegazione: L'IA assegna un punteggio di "sicurezza" a ogni rilevamento.
# Una soglia di 0.5 (50%) è un buon compromesso per questo modello:
# più bassa e vedrai "falsi positivi", più alta e potresti
# perdere rilevamenti validi (specialmente oggetti piccoli).
SOGLIA_CONFIDENZA_DEFAULT = 0.7
SOGLIA_CONFIDENZA_DEBUG = 0.1  # Per mostrare *tutto* quando si preme 'd'


# --- SEZIONE 3: FUNZIONI HELPER ---

def carica_etichette(path):
    """
    Carica le etichette (nomi delle classi) dal file di testo.

    Il file .txt contiene un nome di classe per riga (es. 'person').
    Questa funzione lo mappa a un dizionario (es. {0: 'person', ...}).

    Args:
        path (str): Il percorso al file .txt delle etichette.

    Returns:
        dict: Un dizionario che mappa l'ID (indice) al nome della classe.
        None: Se il file non viene trovato.
    """
    try:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            etichette = {}
            for i, line in enumerate(lines):
                etichette[i] = line.strip()  # .strip() rimuove spazi e a-capi
        print(f"Caricate {len(etichette)} etichette.")
        return etichette
    except FileNotFoundError:
        print(f"ERRORE: File delle etichette non trovato in {path}")
        return None
    except Exception as e:
        print(f"ERRORE critico durante la lettura delle etichette: {e}")
        return None


def disegna_risultati(frame_nativo, oggetti, etichette, scale_x, scale_y):
    """
    Disegna i bounding box e le etichette sul frame nativo.

    Questa è la funzione dove avviene la "magia" del post-processing.
    Utilizza la funzione .scale() di pycoral per mappare
    le coordinate dal frame 300x300 al frame nativo.

    Args:
        frame_nativo (np.array): Il frame video originale (es. 1280x720).
        oggetti (list): La lista di oggetti rilevati da 'detect.get_objects'.
        etichette (dict): Il dizionario {id: nome} caricato in precedenza.
        scale_x (float): Il fattore di scala per la larghezza.
        scale_y (float): Il fattore di scala per l'altezza.

    Returns:
        np.array: Il frame nativo con i disegni sovrimpressi.
    """
    try:
        for obj in oggetti:
            # obj.bbox contiene coordinate relative all'input 300x300
            # (es. xmin=20, ymin=10)

            # 1. LA CORREZIONE CHIAVE (v8):
            # Chiediamo a pycoral di scalare il box per noi usando i
            # nostri fattori di scala (diversi per x e y).
            bbox_scalata = obj.bbox.scale(scale_x, scale_y)

            # 2. CONVERSIONE A PIXEL INTERI
            # Ora bbox_scalata contiene coordinate in pixel nativi (es. 85, 24)
            # Convertiamo in interi per farli digerire a OpenCV.
            xmin = int(bbox_scalata.xmin)
            ymin = int(bbox_scalata.ymin)
            xmax = int(bbox_scalata.xmax)
            ymax = int(bbox_scalata.ymax)

            # 3. CONTROLLO DI SICUREZZA
            # Assicura che le coordinate non siano fuori dai bordi del frame
            if xmin < 0 or ymin < 0 or xmax > (frame_nativo.shape[1]) or ymax > (frame_nativo.shape[0]):
                continue  # Salta questo oggetto

            # 4. PREPARAZIONE TESTO
            nome_classe = etichette.get(obj.id, 'Sconosciuto')
            confidenza = round(obj.score * 100, 1)
            etichetta_testo = f"{nome_classe}: {confidenza}%"

            # 5. DISEGNO del RETTANGOLO (BOX)
            cv2.rectangle(frame_nativo, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # 6. DISEGNO del TESTO
            # Posiziona il testo 10 pixel sopra il box
            y_testo = ymin - 10
            # Se è troppo vicino al bordo, mettilo dentro al box
            if y_testo < 10:
                y_testo = ymin + 20

            cv2.putText(
                frame_nativo, etichetta_testo,
                (xmin, y_testo),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
            )

    except Exception as e:
        # Se qualcosa va storto, stampalo, ma non crashare il loop
        print(f"[ERRORE] nel disegno: {e}")
        pass

    return frame_nativo


def check_prerequisiti():
    """
    Verifica che i file del modello e delle etichette esistano.
    Senza questi file, il programma non può partire.

    Returns:
        bool: True se i file esistono, False altrimenti.
    """
    if not os.path.exists(PATH_MODELLO):
        print(f"ERRORE: File del modello non trovato: {PATH_MODELLO}")
        print("Eseguire 'python setup.py' per scaricare i file necessari.")
        return False

    if not os.path.exists(PATH_ETICHETTE):
        print(f"ERRORE: File delle etichette non trovato: {PATH_ETICHETTE}")
        print("Eseguire 'python setup.py' per scaricare i file necessari.")
        return False

    return True


def trova_risoluzioni_supportate(cap):
    """
    Testa un elenco di risoluzioni standard sulla webcam
    per scoprire quali supporta.

    Args:
        cap (cv2.VideoCapture): L'oggetto webcam già inizializzato.

    Returns:
        list: Una lista di tuple (width, height) supportate.
    """
    print("Ricerca delle risoluzioni supportate dalla webcam...")

    # 1. Ottieni la risoluzione default (è sempre supportata)
    default_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    default_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    default_res = (default_w, default_h)

    # 2. Elenco di risoluzioni standard da provare
    risoluzioni_da_testare = [
        (640, 480),  # VGA (Default comune)
        (800, 600),  # SVGA
        (1024, 768),  # XGA
        (1280, 720),  # 720p HD (16:9)
        (1920, 1080)  # 1080p Full HD (16:9)
    ]

    risoluzioni_supportate = [default_res]

    # 3. Testa ogni risoluzione
    for w, h in risoluzioni_da_testare:
        if (w, h) == default_res:
            continue

        # Prova a impostare la risoluzione
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        # Leggi cosa la webcam ha *effettivamente* accettato
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Se corrisponde, è supportata
        if actual_w == w and actual_h == h:
            if (w, h) not in risoluzioni_supportate:
                risoluzioni_supportate.append((w, h))

    # 4. Ripristina il default per sicurezza
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, default_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, default_h)

    print(f"Risoluzioni trovate: {risoluzioni_supportate}")
    return risoluzioni_supportate


def chiedi_risoluzione_utente(risoluzioni):
    """
    Mostra un menu all'utente nel terminale e restituisce
    la risoluzione scelta (width, height).

    Args:
        risoluzioni (list): La lista di tuple (w, h) supportate.

    Returns:
        tuple: La tupla (width, height) scelta dall'utente.
        None: Se la selezione fallisce o l'utente esce.
    """
    if not risoluzioni:
        print("ERRORE: Nessuna risoluzione trovata.")
        return None

    # Non chiedere se c'è solo un'opzione
    if len(risoluzioni) == 1:
        print(f"Usando l'unica risoluzione trovata: {risoluzioni[0][0]}x{risoluzioni[0][1]}")
        return risoluzioni[0]

    print("\n" + "=" * 40)
    print("--- Scegli la risoluzione della webcam ---")
    default_res = risoluzioni[0]  # Il primo è sempre il default

    for i, (w, h) in enumerate(risoluzioni):
        label = " (Default)" if (w, h) == default_res else ""
        print(f"  [{i + 1}] {w}x{h} {label}")
    print("=" * 40)

    while True:
        try:
            scelta_str = input(f"Inserisci un numero (1-{len(risoluzioni)}) [Default=1]: ")

            # Caso 1: L'utente preme Invio (usa default)
            if not scelta_str:
                return default_res

            # Caso 2: L'utente inserisce un numero
            scelta_int = int(scelta_str)
            if 1 <= scelta_int <= len(risoluzioni):
                return risoluzioni[scelta_int - 1]
            else:
                print(f"Scelta non valida. Inserisci un numero tra 1 e {len(risoluzioni)}.")
        except ValueError:
            print("Input non valido. Inserisci solo un numero.")
        except KeyboardInterrupt:
            print("\nUscita.")
            return None  # Permette al main() di uscire pulitamente


# --- SEZIONE 4: BLOCCO PRINCIPALE (main) ---

def main():
    """
    Il flusso principale del programma.
    Esegue l'inizializzazione, la selezione della webcam e il loop di inferenza.
    """
    print("=" * 60)
    print("--- Progetto Accademico 'Hello Coral' (Versione Webcam) ---")
    print("   Autore: Craicek - v1.0 (Release Documentata)")
    print("=" * 60)
    print()

    # --- 4.1 Controllo Prerequisiti ---
    if not check_prerequisiti():
        print("Esecuzione interrotta. Risolvi gli errori qui sopra.")
        return

    # --- 4.2 Caricamento Etichette ---
    print(f"Caricamento etichette da: {PATH_ETICHETTE}")
    etichette = carica_etichette(PATH_ETICHETTE)
    if etichette is None: return

    # --- 4.3 Caricamento Modello TPU ---
    print(f"Caricamento modello TPU da: {PATH_MODELLO}")
    try:
        # 'make_interpreter' cerca automaticamente la Coral TPU
        # e carica il modello su di essa.
        interpreter = make_interpreter(PATH_MODELLO)
        interpreter.allocate_tensors()  # Alloca la memoria sulla TPU
        print("[OK] Modello e interprete TPU caricati con successo.")
    except Exception as e:
        print(f"[ERRORE] Problema durante il caricamento del modello TPU: {e}")
        print("Verifica che la Coral TPU sia collegata correttamente.")
        print("Su Linux, verifica i permessi 'udev'.")
        return

    # Ottieni la dimensione di INPUT richiesta dal modello (es. 300x300)
    input_size = common.input_size(interpreter)  # Restituisce (width, height)
    print(f"Dimensioni di INPUT richieste dal modello (TPU): {input_size[0]}x{input_size[1]}")
    print()

    # --- 4.4 Inizializzazione e Scelta Webcam ---
    print("Avvio della webcam...")
    cap = cv2.VideoCapture(0)  # 0 = Webcam di default
    if not cap.isOpened():
        print("ERRORE: Impossibile aprire la webcam.")
        return

    # 1. Trova le risoluzioni
    risoluzioni_trovate = trova_risoluzioni_supportate(cap)

    # 2. Chiedi all'utente
    risoluzione_scelta = chiedi_risoluzione_utente(risoluzioni_trovate)
    if risoluzione_scelta is None:
        cap.release()
        return

    # 3. Applica la risoluzione scelta
    w_scelto, h_scelto = risoluzione_scelta
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w_scelto)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h_scelto)

    # 4. Leggi la risoluzione EFFETTIVA (potrebbe essere diversa)
    width_nativo = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_nativo = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"\n[OK] Webcam avviata con risoluzione: {width_nativo}x{height_nativo}")
    print()

    # --- 4.5 Calcolo Fattori di Scala ---
    # Questo è il cuore della logica di "correzione".
    # Lo calcoliamo solo una volta, fuori dal loop, per efficienza.
    scale_x = width_nativo / input_size[0]
    scale_y = height_nativo / input_size[1]

    # Variabili per stato e FPS
    debug_mode = False
    prev_frame_time = 0
    fps = 0

    print("Istruzioni (con la finestra video attiva):")
    print("  • Premi 'q' per USCIRE")
    print("  • Premi 's' per salvare uno SCREENSHOT")
    print("  • Premi 'd' per attivare/disattivare il DEBUG")
    print()

    # --- 4.6 LOOP PRINCIPALE (Inferenza Live) ---
    while cap.isOpened():

        # 1. CALCOLO FPS
        new_frame_time = time.time()

        # 2. LETTURA FRAME NATIVO (es. 1280x720)
        ret, frame_nativo = cap.read()
        if not ret:
            print("Errore: Impossibile leggere il frame dalla webcam.")
            break

        # 3. PRE-PROCESSING (Metodo "Squash")
        #    Converte i colori (OpenCV usa BGR, TFLite vuole RGB)
        frame_rgb = cv2.cvtColor(frame_nativo, cv2.COLOR_BGR2RGB)
        #    "Schiaccia" l'immagine alle dimensioni del modello (300x300)
        frame_schiacciato = cv2.resize(frame_rgb, input_size)

        # 4. ESECUZIONE INFERENZA
        #    Invia il frame 300x300 alla TPU
        common.set_input(interpreter, frame_schiacciato)
        #    Esegue l'inferenza
        interpreter.invoke()

        # 5. POST-PROCESSING (lettura risultati)
        soglia_attuale = SOGLIA_CONFIDENZA_DEBUG if debug_mode else SOGLIA_CONFIDENZA_DEFAULT
        #    Ottiene gli oggetti rilevati (con coordinate 0-300)
        oggetti = detect.get_objects(interpreter, soglia_attuale)

        # 6. DISEGNO RISULTATI
        #    Passiamo i fattori di scala per la mappatura corretta
        frame_con_risultati = disegna_risultati(
            frame_nativo.copy(), oggetti, etichette, scale_x, scale_y
        )

        # 7. VISUALIZZAZIONE INFO (Overlay)
        if prev_frame_time > 0:
            fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time

        # Testi da mostrare
        testo_webcam = f"Webcam (Output): {width_nativo}x{height_nativo}"
        testo_tpu = f"TPU (Input): {input_size[0]}x{input_size[1]} (Squashed)"
        testo_fps = f"FPS: {fps:.1f}"
        testo_oggetti = f"Oggetti: {len(oggetti)}"
        testo_soglia = f"Soglia: {soglia_attuale * 100:.0f}% ({'DEBUG' if debug_mode else 'DEFAULT'})"

        # Disegna lo sfondo nero per il testo
        cv2.rectangle(frame_con_risultati, (0, 0), (450, 110), (0, 0, 0), -1)
        # Disegna i testi
        cv2.putText(frame_con_risultati, testo_webcam, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_con_risultati, testo_tpu, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_con_risultati, testo_fps, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_con_risultati, testo_oggetti, (150, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame_con_risultati, testo_soglia, (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # 8. MOSTRA IL FRAME
        cv2.imshow('Hello Coral - Rilevamento Webcam (Premi "q", "d", "s")', frame_con_risultati)

        # 9. GESTIONE INPUT TASTIERA
        #    cv2.waitKey(1) è fondamentale per permettere a OpenCV
        #    di aggiornare la finestra.
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("Tasto 'q' premuto. Chiusura in corso...")
            break  # Esce dal loop

        elif key == ord('s'):
            filename = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(filename, frame_con_risultati)
            print(f"Screenshot salvato: {filename}")

        elif key == ord('d'):
            debug_mode = not debug_mode
            print(f"Modalità DEBUG {'ATTIVATA' if debug_mode else 'DISATTIVATA'}")

    # --- 4.7 RILASCIO RISORSE ---
    print()
    print("--- Chiusura in corso... ---")
    cap.release()  # Rilascia la webcam
    cv2.destroyAllWindows()  # Chiude tutte le finestre OpenCV
    print("Esecuzione terminata con successo.")


# Questo è un costrutto standard in Python.
# Significa: "esegui la funzione main() solo se questo
# script è stato lanciato direttamente (non importato come modulo)".
if __name__ == "__main__":
    main()