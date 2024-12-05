# quadric-canonicalizer
A program that brings a quadric surface in canonical form and renders the transformation.
## Introduzione

Il programma si occupa di portare in forma canonica metrica una superficie quadrica e renderizzare la trasformazione.
La matematica dietro all'algoritmo e' stata attinta da 3 fonti principali, vedere la bibliografia: \ref{bibliografia}. 

### Notazione decisa

Sia una quadrica una funzione il luogo dei punti che soddisfano $f(x, y, z)=0$ ossia gli zeri di un polinomio di secondo grado in 3 variabili $f(x, y, z)$.
Definiamo: $\underline x = \begin{pmatrix} x \\ y \\ z \end{pmatrix}$ e il suo "esteso" $\underline {\overline x} = \begin{pmatrix} x \\ y \\ z \\ 1 \end{pmatrix}$.
Una quadrica $q$ puo' essere identificata univocamente dalla sua matrice dei coefficienti estesa  $\overline A \in Mat_4(\mathbb{R})$
$$\overline  A = \begin{pmatrix}
A & \underline b \\
\underline b^t & \delta=a_{4,4}
\end{pmatrix}$$
con la seguente espressione
$$\Gamma : \underline {\overline x} ^t \cdot \overline A \cdot \underline {\overline x} = 0 \iff \underline x ^t \cdot A \cdot \underline x + 2 \cdot \underline b ^t \cdot \underline x + a_{4,4}=0$$
dove $A \in Mat_3(\mathbb{R})$ che identifica la forma quadratica $q(\underline x) = \underline x ^ t A \underline x$ associata alla quadrica. A contiene i termini quadratici sulla diagonale e i termini rettangolari, e $\underline b \in \mathbb{R}^3$ contiene i coefficienti dei termini lineari.
In generale il risultato fondamentale ("teorema di classificazione") e' che una quadrica puo' essere ridotta in una tra 3 delle seguenti forme canoniche metriche: $$\sum_{i=1}^r c_i z_i^2 \quad \text{se } \text{rk}(\overline A) = \text{rk}(A)$$
$$\sum_{i=1}^r c_i z_i^2 + c_{r+1} \quad \text{con } c_{r+1} \neq 0 \text{ se } \text{rk}(\overline A) = \text{rk}(A)+1$$
$$\sum_{i=1}^r c_i z_i^2 + 2c_{r+1}z_{r+1} \quad \text{con } c_{r+1} \neq 0 \text{ se } \text{rk}(\overline A) = \text{rk}(A)+2$$ tramite un'isometria.Le forme canoniche metriche sono uniche, data una quadrica (non vuota, per la precisione), a meno di una moltiplicazione dell'equazione per uno scalare e a meno di una permutazione delle indeterminate. Nel nostro caso l'isometria e' identificabile con una matrice a blocchi $$P  = \begin{pmatrix} S & \underline c\\\  \underline 0^t & 1 \end{pmatrix} \in \text{Mat}_4(\mathbb{R})$$ oppure si scrive come sistema lineare $$\underline x = S \underline x' + \underline c$$ dove $S \in \text{Mat}_3(\mathbb{R})$ e' un'isometria con $\det = \pm 1$ (rotazione nel caso $\det S =1$) e $\underline c \in \mathbb{R}^3$ e' il vettore di traslazione.

## Il programma

Dividiamo il programma in "main", "parte numerica" e "parte grafica".

### Main

Il main riceve in input l'equazione della quadrica, il filepath (percorso dove inserire i file video renderizzati da manim) e la qualita' desiderata dell'output. Il main esegue prima la parte numerica (canonize\_quadric in transformer.py) per portare la quadrica in forma canonica e ottenere tutti le informazioni necessarie alla parte grafica, e poi renderizza la trasformazione (scene\_render.py)

### Parte numerica

#### Funzionamento

La parte numerica ha il compito di trasformare la quadrica conducendola in forma canonica tramite una traslazione e una rotazione. Al momento le due trasformazioni possono essere eseguite in due ordini:
- nel caso delle quadriche a centro e del cilindro parabolico viene eseguita prima la traslazione e poi la rotazione.
- nelle altre quadriche non a centro viceversa.
L'ordine e' importante poiche' si riflette nella visualizzazione della trasformazione delle quadrica.
La parte numerica salva, in un dizionario che viene ritornato alla parte grafica, tutte le informazioni necessarie a quest'ultima: un vettore di traslazione e una matrice di rotazione, la matrice $\overline A$ (in tre versioni: originaria, dopo la prima trasformazione e dopo la seconda), le tre equazioni della quadrica corrispondenti alle tre versioni di $\overline A$ e il tipo di quadrica.
Il processo per condurre in forma canonica e' descritto nel dettaglio, da un punto di vista matematico, in \ref{trasformare la quadrica}

#### Moduli

La parte numerica e' divisa nei seguenti moduli:
- transformer.py: contiene la funzione "canonize\_quadric" che prende come input una stringa rappresentante l'equazione della quadrica e outputta un dictionary contenente varie informazioni sulla quadrica e sulle sue trasformazioni. Contiene le varie funzioni di supporto che si occupano di portare la quadrica in forma canonica metrica e calcolare le trasformazioni necessarie a farlo.
- tester.py: contiene "test\_quadrics", una funzione che genera quadriche, per ogni tipo di quadrica desiderato, e controllano che venga classificata correttamente. Per adesso non viene controllato che l'espressione/la matrice $\overline A$ della quadrica in forma canonica metrica (outputtata da "canonize\_quadrics") contenga effettivamente solo i termini che ci si aspetta dal teorema di classificazione. Vedere \ref{testing} per dettagli.
- checker.py: contiene funzioni che servono a fare vari controlli di correttezza dell'algoritmo.
- parabolic\_cylinder.py: contiene la funzione parabolic\_cylinder\_canonize che calcola le trasformazioni necessarie a condurre un cilindro parabolico in forma canonica metrica, con le varie funzioni di supporto. Si trova in un modulo separato in quanto e' calcolata diversamente. 
- misc: varie funzioni di supporto ad altre funzioni e di conversione di stringhe in espressioni sympy.
- classifier.py: contiene una funzione "classify\_quadric" che ritorna un intero che classifica la quadrica, a partire dalle sue matrici (ogni intero e' mappato a un tipo di quadrica secondo il dictionary ENUM\_QUADRICS). Per farlo a partire dalla stringa contenente l'equazione, usare "expr2classification".

#### Testing

Per dettagli vedere il modulo "tester.py"
Il testing della parte numerica e' stato eseguito generando randomicamente delle quadriche di cui conosciamo il tipo e vedere:
- se la quadrica e' classificata correttamente, che e' possibile controllare in modo totalmente automatizzato in "tester.py".
- se la forma canonica metrica "rispecchia" la forma canonica che ci si aspetta (ad esempio un ellissoide non dovrebbe contenere un termine rettangolare o lineare in forma canonica metrica) - questo lato (chiamiamolo "controllo per termini") non e' ancora stato automatizzato in "tester.py" ma e' stato fatto manualmente.
E' stato eseguito testing per ogni tipo di quadrica, anche se non nel modo piu' generale possibile, inserendo dei coefficienti che introducono rotazioni/traslazioni in modo randomico a partire dalle forme canoniche metriche implicite.
Non si sono (per ora) presentati errori, tranne in un caso, il cilindro parabolico.
Infatti capita che esso non sia condotto in forma canonica metrica correttamente, probabilmente a causa di rounding error ripetuti dovuti a sympy (che fa in generale piu' fatica a fare il tipo di calcoli richiesti in modo accurato).
Si potrebbe testare totalmente randomicamente (ossia a partire da un polinomio generico di secondo grado con coefficienti random) una volta automatizzato il controllo "per termini" menzionato sopra.

#### Cose da migliorare
Ricapitoliamo cose da migliorare nel codice:
- attualmente non e' implementato un modo per distinguere i cilindri ellittici reali da quelli complessi e i piani paralleli reali da quelli complessi
- il cilindro parabolico presenta dei casi che non vengono portati in forma canonica correttamente. Si potrebbe risolvere in due modi:
\begin{itemize}
    - Usare un metodo che evita risoluzioni simboliche.
    - Assicurarsi una miglior gestione dei rounding errors dovuti all'aritmetica floating point.
\end{itemize}
- implementare un modo di controllo di errore che guardano i coefficienti dei termini
- implementare un metodo robusto di tolleranza errori in aritmetica floating point.
- rendere tutte le matrici con $\det S=1$, ossia permutare quelle con $\det S=-1$ (e le righe corrispettive in $D$)
- opzionale, solo se serve generalizzare a iperquadriche: usare matrici di permutazione che riconducono una quadrica generica a una permutazione specifica di indeterminate, anziche' i vari if-else per ogni possibile permutazione.

### Parte grafica

La parte grafica renderizza la trasformazione della quadrica da forma originale a forma canonica in manim.
Attualmente supporta tutte le quadriche tranne quelle immaginarie.

#### Funzionamento
Dopo aver ottenuto tutto il necessario (il dictionary precedentemente citato) dalla parte numerica, la parte grafica funziona nel modo seguente:
1. manim necessita di funzioni in forma parametrica per funzionare e non accetta equazioni implicite. Dunque, partendo dalla equazione canonica metrica implicita, otteniamo la forma parametrica facilmente (vedere \ref{forme parametriche}), e applichiamo le trasformazioni all'inverso in modo da ottenere la quadrica originaria. Infatti cio' ci consente di ottenere la forma parametrica della quadrica originaria senza dover applicare metodi generali di parametrizzazione, che sarebbero piu' difficili da implementare. Questa parte non viene renderizzata. In certi casi (es. iperboloide a due falde) viene generata parametricamente solo una falda, e l'altra e' ottenuta con una riflessione della prima falda rispetto al centro.
2. La quadrica viene condotta in forma canonica applicando le trasformazioni correttamente, e viene renderizzata la trasformazione (vedere ordine delle trasformazioni nella parte numerica).
\end{enumerate}

#### Moduli

- scene\_render.py: coordina le varie trasformazioni della quadrica, ossia le fasi della scena. Setta gli assi e la posizione della telecamera.
- create\text\_overlay.py: crea gli oggetti di testo mostrati sopra alla trasformazione della quadrica, e li passa a scene\_render.py.
- create\_quadric\_surface.py: crea l'oggetto quadrica iniziale in forma canonica (inizialmente non renderizzato, vedere funzionamento).

#### Testing

Non ancora eseguito.

#### Cose da migliorare

- fare un po' di testing sui limiti della parte grafica
- testare resolution in funzione dei coefficienti per quadriche "grandi"
- testare range (dei punti) "modulari"/adattabili per rappresentare bene le quadriche in funzione dei coefficienti - vedi anche resolution
