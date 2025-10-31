# Installazione Google Coral USB Edge TPU su Windows

Questa guida ti permette di installare la Google Coral USB Edge TPU su PC Windows, inclusi i driver necessari e le dipendenze fondamentali.

- ([Link Ufiiciale:](https://gweb-coral-full.uc.r.appspot.com/docs/accelerator/get-started/#runtime-on-windows))

## Requisiti

- Sistema operativo: Windows 10 o superiore
- Porta USB 3.0 disponibile
- Microsoft Visual C++ Redistributable 2019 ([Download](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist))

## Istruzioni

### 1. Installa Microsoft Visual C++ Redistributable

Scarica e installa il runtime Visual C++ più aggiornato utilizzando [questo link ufficiale](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist).

### 2. Scarica i driver Coral USB TPU

Vai alla pagina delle release coral su GitHub:

[https://github.com/google-coral/edgetpu/releases](https://github.com/google-coral/libedgetpu/releases)
Scarica la release più recente per **Windows** (`edgetpu_runtime_<version>_windows.zip`).

### 3. Installa i driver TPU

- Estrai il contenuto del file ZIP scaricato.
- Esegui il file `install.bat` come amministratore (clicca destro > "Esegui come amministratore").
- Attendi la fine del processo di installazione e segui eventuali istruzioni aggiuntive a schermo.

### 4. Collega Coral TPU

- Collega la Coral USB Edge TPU ad una porta USB 3.0 del tuo PC.
- Riavvia il computer se richiesto.

## Verifica installazione

Apri un terminale (cmd o Powershell) e, dalla directory runtime estratta, esegui:

edgetpu_runtime.exe --version

Dovresti vedere la versione del runtime installata correttamente.

## Link utili

- [Guida ufficiale Coral USB TPU](https://coral.ai/docs/accelerator/get-started)
- [Edgetpu Repository GitHub](https://github.com/google-coral/edgetpu/releases)
- [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist)
- [Guida dettagliata Windows](https://johngalea.wordpress.com/2024/06/28/coral-tpu-on-windows/)

## Troubleshooting

Consulta [questa guida blog](https://johngalea.wordpress.com/2024/06/28/coral-tpu-on-windows/) per suggerimenti e risoluzione problemi più comuni in fase di installazione.

---

**Autore:**  
Craicek, aggiornata a ottobre 2025
