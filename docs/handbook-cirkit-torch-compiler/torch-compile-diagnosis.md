# Diagnosi della proposta di compilazione intermedia

Verifica del 11 settembre 2026 sul checkout locale `quad-spn`, HEAD `6db3074`.
Documento esaminato: [intermediate-jit-compiler.md](intermediate-jit-compiler.md).

## Esito

La descrizione del backend come interprete Python di un DAG e' corretta.
Non e' invece dimostrata un'impossibilita' generale di usare `torch.compile`:
12 configurazioni piccole del backend attuale sono state catturate interamente
senza modificare la libreria. Il documento originale parla prudentemente di
possibili graph break; queste possibilita' non vanno presentate come cause gia'
accertate per il forward ordinario.

Un compilatore intermedio resta una proposta di ottimizzazione da misurare,
non un prerequisito dimostrato per accedere a TorchDynamo.

## Ambiente e metodo

- Windows, Python 3.12.7, PyTorch 2.12.0+cpu; CUDA non disponibile.
- `torch._dynamo.explain` applicato alla somma di `circuit(x) - partition()`.
- `torch.compile(..., backend="aot_eager", fullgraph=True)` sullo stesso calcolo.
- Confronto del risultato e dei gradienti rispetto ai parametri con eager,
  tolleranze `rtol=1e-4`, `atol=1e-5`.
- Secondo confronto forward con input diverso e batch da 4 a 3 elementi.
  Questo verifica correttezza al cambio di batch, non assenza di ricompilazioni.
- La funzione di partizione e' costruita con `SF.integrate` e compilata con lo
  stesso `TorchCompiler`, come nel percorso di `dataset_eval.py`.

| Casi | Configurazioni | Risultato |
| --- | --- | --- |
| Circuito categorico strutturato | folding on/off, ottimizzazione on/off, sum-product/lse-sum: 8 combinazioni | 1 grafo, 0 graph break; forward e gradienti corretti |
| Circuito gaussiano | folding e ottimizzazione on, lse-sum | Come sopra |
| QuadGraph 3x3 CP | K=3, folding e ottimizzazione on, lse-sum | Come sopra |
| QuadGraph 3x3 Tucker | K=3, folding e ottimizzazione on, lse-sum | Come sopra |
| QuadGraph RGB 3x3 con condivisione dei parametri delle foglie | 256 categorie, mixing weights, K=3, folding e ottimizzazione on, lse-sum | Come sopra |

Il caso RGB ricostruisce direttamente il circuito simbolico del baseline
`LearnSPN` con `input_sharing="full"` e `use_estimated_weights=False`.
Non e' una chiamata a `LearnSPN`: l'ambiente non ha `fast_pytorch_kmeans`.
Gli input sintetici categorici usano valori 0/1 validi; non e' un test del
dataset CelebA completo, della stima informata, o dell'intero training loop.

La prima costruzione dei due QuadGraph utilizzava i pesi di default, che possono
essere negativi: produceva NaN anche in eager. Il test e' stato corretto usando
pesi uniformi positivi. I risultati sopra sono quelli delle prove corrette.

Script: [diagnose_torch_compile.py](../../scripts/diagnose_torch_compile.py).
Risultati: [torch_compile_diagnostic_results.json](../../scripts/torch_compile_diagnostic_results.json).

```bash
python -m scripts.diagnose_torch_compile --backend aot_eager
python -m scripts.diagnose_torch_compile --backend inductor --quick
```

`aot_eager` verifica cattura Dynamo e AOTAutograd. Non genera i kernel ottimizzati
di Inductor. La prova Inductor locale fallisce con `Compiler: cl is not found`:
manca il compilatore C++ nell'ambiente. Questo non identifica un difetto del
circuito e non permette di concludere nulla sullo speedup GPU.

## Verifica delle motivazioni nel documento

### Interprete, generatori, liste e selezione dei moduli

[TorchDiAcyclicGraph.evaluate](../../cirkit/backend/torch/graph/modules.py)
itera effettivamente su `address_book.lookup`, esegue `module(*inputs)` e fa
`module_outputs.append(y)`. `AddressBook.__iter__` produce dataclass e usa
`getattr` per gli indici. Anche
[TorchParameter.forward](../../cirkit/backend/torch/parameters/parameter.py)
chiama lo stesso interprete.

Sono costi reali in eager, ma nelle prove Dynamo attraversa questa logica durante
la cattura e registra un unico grafo tensoriale. La topologia, i tipi dei moduli
e gli identificatori degli archi non dipendono dai valori del batch. La semplice
presenza di Python, generatori o liste non dimostra un graph break.

### Topologia e indici del folding

L'address book viene gia' costruito nel costruttore di `TorchDiAcyclicGraph`.
Gli indici tensoriali sono gia' registrati come buffer da `AddressBook.__init__`.
`LayerAddressBook.lookup` esegue gather, concatenazioni e indicizzazioni, ma non
ricostruisce la topologia a ogni batch. Bisogna distinguere un indice tensoriale
da un ramo Python dipendente dai dati: il primo non e' automaticamente un
ostacolo alla cattura. Il folding passa nei casi provati.

### Semiring, einsum e autograd personalizzato

Il semiring viene scelto una volta nel costruttore di `TorchCompiler`.
Le chiamate ai classmethod in [semiring.py](../../cirkit/backend/torch/semiring.py)
possono essere attraversate da Dynamo. Passano sia CP (`TorchCPTLayer`) sia
Tucker (`TorchTuckerLayer`), incluso il relativo percorso `einsum`.

`LSESumSemiring.apply_reduce` utilizza `SafeLog`; i confronti AOTAutograd dei
gradienti passano. Quindi `autograd.Function` non e' un blocco generale in questo
percorso. `ComplexSafeLog` e i circuiti complessi non sono stati verificati.

Specializzare le operazioni puo' ancora migliorare tempi di tracing, numero di
guardie, uso della memoria o kernel prodotti. Questi benefici richiedono benchmark
separati: zero graph break non significa automaticamente esecuzione piu' veloce.

## Ostacoli concreti trovati

### Query di marginalizzazione a runtime

In [queries.py](../../cirkit/backend/torch/queries.py), `IntegrateQuery._layer_fn`
contiene alla riga 136:

```python
if not torch.any(integration_mask).item():
    return output
```

La prova eager funziona; `fullgraph=True` fallisce proprio su questo ramo con
`Could not guard on data-dependent expression`. Estrarre lo scalare e poi usarlo
in un `if` richiede una decisione Python basata sui valori del tensore.

Una correzione candidata per le foglie supportate e' calcolare sempre
`layer.integrate()` e selezionare con il `torch.where` gia' presente, eliminando
il ritorno anticipato. Richiede una verifica dedicata di semantica e prestazioni;
non e' stata applicata alla libreria in questa diagnosi.

Questa query NON e' la funzione di partizione di `dataset_eval.py`: lo script
costruisce il circuito integrato simbolicamente prima del training. Quel percorso
e' incluso nelle prove riuscite.

Altri rami dipendenti dai valori si trovano nelle conversioni complex-to-real
(`if torch.all(torch.isreal(x))` in `semiring.py`) e nelle validazioni dei pesi
nel sampling. Sono candidati da verificare, non fallimenti riprodotti qui.

### TorchScript e torch.compile sono percorsi diversi

`torch.jit.script(circuit)` fallisce nelle prove con
`Comprehension ifs are not supported yet`, nella proprieta' `inputs` di
`cirkit/utils/algorithms.py`, riga 148:

```python
return (n for n in self._nodes if not self.node_inputs(n))
```

E' il primo ostacolo osservato del frontend TorchScript. Non dimostra che sia
l'unico, ne' si applica automaticamente a TorchDynamo, che passa le prove sopra.
La proposta dovrebbe nominare esplicitamente `torch.compile` come destinazione.

### Stato dello script di training al momento della diagnosi

Nella versione analizzata di [dataset_eval.py](../../cirkit/dataset_eval.py),
righe 524-525, `ctx.compile`
traduce il circuito simbolico nel backend Cirkit/PyTorch; non invoca
`torch.compile`. In quella versione non c'era una chiamata a quest'ultimo nel file.

Il primo esperimento sul workload reale dovrebbe compilare il calcolo tensoriale
di likelihood e partizione, lasciando caricamento dati, logging, `.item()` delle
metriche e salvataggi fuori dalla funzione. Il primo forward va effettivamente
eseguito: la sola creazione del wrapper `torch.compile` non prova la compilazione.

### Integrazione successiva: eseguire i test GPU

`dataset_eval.py` ora accetta `--torch-compile`. Compila la NLL media
`-(circuit(batch) - partition_function()).mean()` e usa lo stesso callable in
training, validazione e test. I moduli originali restano gli oggetti usati
dall'optimizer e dai checkpoint; il file dei pesi mantiene le chiavi originali.
Senza il flag viene eseguita la stessa loss in eager.

Esempio dalla directory contenente `dataset_eval.py`, con i tensori gia' preparati:

```bash
python dataset_eval.py --dataset=celeba --root=/content/celeba64 \
    --rg=quad-graph --inner-layer=cp --k=32 --input-sharing=full \
    --ycc=lossy --activation=clamp --weights-init=uniform \
    --adaptive-alpha --use-mixing-weights \
    --torch-compile --compile-fullgraph --save-path=celeba_compiled.pt
```

- `--compile-backend`: `inductor` (default), `eager` o `aot_eager`.
- `--compile-mode`: `default`, `reduce-overhead`, `max-autotune`,
  `max-autotune-no-cudagraphs`. Le modalita' non default richiedono Inductor.
- `--compile-fullgraph`: fa fallire la cattura se incontra graph break.
  Senza questo flag PyTorch puo' compilare regioni separate.

Per il confronto eager, rimuovere `--torch-compile` e usare un altro
`--save-path`, mantenendo uguali seed, dati e iperparametri. Separare il costo
iniziale della compilazione dalle iterazioni successive; la prima validazione
puo' compilare una variante per inference. Nessuno speedup GPU e' stato misurato
nell'ambiente locale CPU.

Verifica locale della nuova integrazione:

```bash
python -m unittest tests.test_dataset_eval_compile -v
```

Il test usa circuiti reali e `aot_eager/fullgraph`: controlla loss e gradienti,
aggiornamenti Adam, alternanza train/validation, batch finale ridotto,
salvataggio/ripristino dei pesi e valutazione con il callable gia' compilato.

## Correzioni necessarie alla proposta di IR

1. Presentare static codegen come ipotesi di ottimizzazione, da confrontare con
   il backend attuale piu' `torch.compile`, non come requisito di compatibilita'.
2. Registrare i graph break effettivi per versione, circuito e backend prima di
   decidere quali astrazioni riscrivere.
3. Riconoscere che topologia, scelta del semiring e buffer degli indici sono gia'
   preparati prima del forward.
4. Preservare le convenzioni numeriche: `LSESumSemiring.prod` non e' soltanto
   `torch.sum`, ma aggiunge `clamp_min(LOG_CLAMP_MIN)`. `SafeLog.backward` usa
   `nan_to_num`. Sostituire tutto con sum/log/logsumexp standard puo' cambiare
   output e gradienti nei casi limite.
5. Misurare compile time, ricompilazioni, memoria, latenza dopo warm-up e backward
   su CPU/GPU e circuiti realistici. I tempi registrati dallo script includono
   explain e compilazione: non sono benchmark di velocita'.

Riferimento sul metodo: la [guida ufficiale PyTorch al troubleshooting](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
distingue le prove con backend `eager`, `aot_eager` e `inductor`, e descrive
`fullgraph=True` e i rami dipendenti dai dati per localizzare i problemi.
