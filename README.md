# Progetto "Hello Coral" (Google Coral TPU USB)
**Autore:** Craicek

Benvenuti in questo progetto accademico! L'obiettivo è fornire un'implementazione semplice, robusta e ampiamente documentata per eseguire il rilevamento di oggetti (Object Detection) in tempo reale utilizzando una Webcam e un acceleratore Google Coral TPU USB.

Questo non è solo un "copia-incolla", ma uno strumento didattico. Il codice è interamente commentato per spiegare perché vengono fatte certe scelte, in particolare come risolvere il problema più comune di tutti: il conflitto tra la risoluzione della webcam e la risoluzione fissa del modello AI.

## 🧭 Indice
- [Demo del Progetto](#-demo-del-progetto)
- [Il Concetto Chiave: Il Problema del 300x300](#-il-concetto-chiave-il-problema-del-300x300)
- [Funzionalità Principali](#-funzionalità-principali)
- [Guida Rapida: Installazione e Avvio](#-guida-rapida-installazione-e-avvio)
- [Analisi Dettagliata del Codice](#-analisi-dettagliata-del-codice)
- [Risorse e Link Ufficiali](#-risorse-e-link-ufficiali)

## 🖼️ Demo del Progetto
Il programma avviato rileva la tua webcam, ti chiede di scegliere una risoluzione (anche Full HD) e poi apre una finestra di OpenCV.

Il testo in sovrimpressione (overlay) mostra:

- **Webcam (Output):** La risoluzione nativa che hai scelto (es. 1280x720).
- **TPU (Input):** La risoluzione fissa richiesta dal modello (300x300).
- **FPS:** I fotogrammi al secondo (un indicatore di performance).
- **Oggetti:** Il numero di oggetti rilevati nel frame.
- **Soglia:** La soglia di confidenza attuale (modificabile con il tasto 'd').

Grazie alla logica di correzione, i box verdi si adattano perfettamente agli oggetti, nonostante l'immagine venga "schiacciata" per l'analisi.

(Suggerimento: Registra un breve video o GIF del programma in azione e inseriscilo qui!)
![Demo del Progetto](link_alla_tua_demo.gif)

## 💡 Il Concetto Chiave: Il Problema del 300x300
Questo è il cuore accademico del progetto.

### 1. Il Conflitto
- La tua **Webcam** è ad alta risoluzione e ha proporzioni (aspect ratio) 16:9 (es. 1280x720) o 4:3 (es. 640x480).
- Il **Modello AI** (MobileNet SSD v2) è stato addestrato per essere velocissimo. Per farlo, accetta solo un input fisso e quadrato: 300x300 (proporzioni 1:1).

### 2. La Soluzione (Metodo "Squash & Scale")
Ci sono due modi per risolvere questo: "Letterbox" (imbottitura) o "Squash" (schiacciamento). Abbiamo scelto lo Squash perché è più semplice e veloce, fedele al principio "Semplice vince su complicato".

**PRE-PROCESSING (Squash):** Prendiamo il frame nativo (es. 640x480) e lo "schiacciamo" brutalmente a 300x300 con `cv2.resize`. L'immagine inviata alla TPU è distorta.

**INFERENZA:** La TPU analizza l'immagine distorta e ci dice: "Ho trovato una 'persona' alle coordinate (X=50, Y=100) dell'immagine 300x300".

**POST-PROCESSING (Scale):** Qui sta la magia. Se disegnassimo quel box, sarebbe disallineato. Dobbiamo "annullare" la distorsione. Calcoliamo due fattori di scala separati:

```
scale_x = width_nativo / 300 (es. 640 / 300 = 2.13)
scale_y = height_nativo / 300 (es. 480 / 300 = 1.6)
```

Usiamo poi la funzione `bbox.scale(scale_x, scale_y)` della libreria pycoral. Questa funzione moltiplica le coordinate X del box per 2.13 e quelle Y per 1.6, mappando perfettamente il box distorto sul frame originale non distorto.

Questo approccio ci permette di usare qualsiasi risoluzione della webcam in modo automatico.

## ✨ Funzionalità Principali
- **Rilevamento Risoluzione:** All'avvio, testa la tua webcam e ti presenta un menu per scegliere la risoluzione da usare.
- **Correzione Proporzioni (Scaling):** Implementa la logica "Squash & Scale" per mappare accuratamente i box, anche con proporzioni diverse.
- **Script di Setup "1-Click":** Il file setup.py gestisce tutto: controlla il venv, installa le dipendenze e scarica i modelli.
- **Modalità Debug Interattiva:** Premi 'd' per abbassare la soglia di confidenza e vedere tutti i rilevamenti, anche quelli a bassa probabilità.
- **Codice Documentato:** L'intero main.py è commentato linea per linea per scopi didattici.

## 🚀 Guida Rapida: Installazione e Avvio
Segui questi 5 passi per avere il progetto funzionante.

### Passo 1: Prerequisiti Hardware e Software
**Hardware:** Una Google Coral TPU USB e una Webcam.

**Software:** Python 3.8+ e (su Windows) i driver della Coral.

**Runtime di Sistema:** Devi installare la runtime della Edge TPU.

**Su Linux (Debian/Ubuntu/Raspberry Pi):**
```bash
curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo gpg --dearmor -o /usr/share/keyrings/coral-edgetpu-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/coral-edgetpu-archive-keyring.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install libedgetpu1-std
```

**Su Windows 10/11:** Devi installare manualmente sia il Driver che la Runtime. Segui la guida ufficiale qui: 🔗 [Guida Installazione Windows](https://coral.ai/docs/accelerator/get-started/#windows)

### Passo 2: Clona il Repository
Usa Git per clonare il progetto sul tuo computer:
```bash
git clone https://github.com/BitMakerMan/GoogleCoralTpuUSB.git
cd GoogleCoralTpuUSB
```

### Passo 3: Crea e Attiva un Ambiente Virtuale (VENV)
Questo è fondamentale per non "sporcare" il tuo Python di sistema.
```bash
# Crea un ambiente virtuale chiamato 'venv'
python -m venv venv

# Attiva l'ambiente
# Su Windows (cmd.exe):
.\venv\Scripts\activate
# Su Linux/Mac/Git Bash:
source venv/bin/activate
```
Vedrai `(venv)` apparire all'inizio della riga del tuo terminale.

### Passo 4: Esegui lo Script di Setup
Questo è lo script magico. Con il venv attivo, esegui:
```bash
python setup.py
```
Lo script farà tre cose:
1. Verificherà che sei in un venv.
2. Installerà tutte le dipendenze da requirements.txt.
3. Creerà la cartella /modelli e scaricherà il modello .tflite e le etichette .txt.

### Passo 5: Avvia il Programma!
Sei pronto. Connetti la tua Coral TPU e la webcam, e lancia il programma:
```bash
python main.py
```
Segui le istruzioni nel terminale per scegliere la risoluzione e goditi il rilevamento!

### Controlli da Tastiera
Mentre la finestra di OpenCV è attiva:
- **q** : Chiude il programma.
- **d** : Attiva/Disattiva la Modalità Debug (soglia di confidenza 10%).
- **s** : Salva uno Screenshot del frame corrente nella cartella del progetto.

## 🔬 Analisi Dettagliata del Codice
Il progetto è diviso in tre file principali, tutti documentati internamente.

### requirements.txt
Contiene la lista dei pacchetti Python. Due righe sono fondamentali:
- `--extra-index-url ...`: Dice a pip di cercare anche nel repository speciale di Google, dove si trovano pycoral e tflite-runtime.
- `numpy<2`: Forza l'installazione di NumPy v1.x, dato che (a fine 2025) pycoral non è ancora compatibile con NumPy v2.0+, evitando un crash all'avvio.

### setup.py
Lo script di installazione automatizzato. Il suo main() è diviso in 3 fasi:
- `check_venv()`: Impedisce l'avvio se non rileva un ambiente virtuale attivo.
- `installa_dipendenze()`: Usa subprocess per eseguire `pip install -r requirements.txt` in modo sicuro.
- `scarica_file_modello()`: Usa urllib per scaricare i file del modello e delle etichette dagli URL ufficiali di Google.

### main.py
Il cuore del progetto. È diviso in 4 sezioni logiche:

**Sezione 1: Importazioni**
- `cv2` (OpenCV) per tutto ciò che è video.
- `pycoral` per parlare con la TPU.
- `time`, `os`, `sys` per utilità (FPS, file, menu).

**Sezione 2: Costanti**
- Percorsi ai file e soglie di confidenza.

**Sezione 3: Funzioni Helper**
- `carica_etichette()`: Legge il file .txt e lo trasforma in un dizionario.
- `trova_risoluzioni_supportate()` e `chiedi_risoluzione_utente()`: Gestiscono la logica del menu interattivo all'avvio.
- `disegna_risultati(...)`: La funzione più importante. Riceve i box (da 0-300), li scala con `bbox.scale(scale_x, scale_y)` e li disegna sul frame nativo.

**Sezione 4: main()**
1. **Inizializzazione:** Carica etichette e modello.
2. **Setup Webcam:** Chiama le funzioni helper per il menu e imposta la risoluzione scelta (`cap.set(...)`).
3. **Calcolo Scala:** Calcola `scale_x` e `scale_y` una sola volta, fuori dal loop.
4. **Loop Principale:** Il ciclo `while True` che esegue la pipeline a ogni frame:
   - `cap.read()` (Leggi frame 1280x720)
   - `cv2.resize()` (Pre-processa a 300x300)
   - `interpreter.invoke()` (Inferenza sulla TPU)
   - `detect.get_objects()` (Ottieni risultati)
   - `disegna_risultati()` (Post-processa e disegna)
   - `cv2.imshow()` (Mostra il frame)
5. **Rilascio:** `cap.release()` e `cv2.destroyAllWindows()` per una chiusura pulita.

## 🔗 Risorse e Link Ufficiali
- **Repository GitHub:** https://github.com/BitMakerMan/GoogleCoralTpuUSB
- **Sito Ufficiale Google Coral:** https://coral.ai/
- **Modello Utilizzato:** https://dl.google.com/coral/canned_models/mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite
- **Etichette Utilizzate:** https://raw.githubusercontent.com/google-coral/test_data/master/coco_labels.txt
- **Guida Ufficiale Installazione (Windows):** https://coral.ai/docs/accelerator/get-started/#windows

## 📜 Licenza
Questo progetto è rilasciato sotto la Licenza MIT. Sei libero di usare, modificare e distribuire questo codice per qualsiasi scopo, accademico o commerciale.

(Sentiti libero di aggiungere un file LICENSE con il testo della licenza MIT nel tuo repository)
