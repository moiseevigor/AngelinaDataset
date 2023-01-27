# import the braille library
from PIL import Image, ImageDraw, ImageFont

# wrong encodeing !!!!
braille_map = {
    'a': '⠁', 'b': '⠃', 'c': '⠉', 'd': '⠙', 'e': '⠑', 'f': '⠋',
    'g': '⠛', 'h': '⠓', 'i': '⠊', 'j': '⠚', 'k': '⠅', 'l': '⠇',
    'm': '⠍', 'n': '⠝', 'o': '⠕', 'p': '⠏', 'q': '⠟', 'r': '⠗',
    's': '⠎', 't': '⠞', 'u': '⠥', 'v': '⠧', 'w': '⠺', 'x': '⠭',
    'y': '⠽', 'z': '⠵', ' ': '⠀',
    'à': '⠠⠁', 'è': '⠠⠑', 'é': '⠠⠑', 'ì': '⠠⠊', 'ò': '⠠⠕', 'ù': '⠠⠥',
    'A': '⠠⠁', 'B': '⠠⠃', 'C': '⠠⠉', 'D': '⠠⠙', 'E': '⠠⠑', 'F': '⠠⠋',
    'G': '⠠⠛', 'H': '⠠⠓', 'I': '⠠⠊', 'J': '⠠⠚', 'K': '⠠⠅', 'L': '⠠⠇',
    'M': '⠠⠍', 'N': '⠠⠝', 'O': '⠠⠕', 'P': '⠠⠏', 'Q': '⠠⠟', 'R': '⠠⠗',
    'S': '⠠⠎', 'T': '⠠⠞', 'U': '⠠⠥', 'V': '⠠⠧', 'W': '⠠⠺', 'X': '⠠⠭',
    'Y': '⠠⠽', 'Z': '⠠⠵',
    '1': '⠼⠁', '2': '⠼⠃', '3': '⠼⠉', '4': '⠼⠙', '5': '⠼⠑', '6': '⠼⠋',
    '7': '⠼⠛', '8': '⠼⠓', '9': '⠼⠊', '0': '⠼⠚',
    '!': '⠈⠁', '"': '⠈⠃', '#': '⠈⠉', '$': '⠈⠙', '%': '⠈⠑', '&': '⠈⠋', 
    '(': '⠈⠛', ')': '⠈⠓', '*': '⠈⠊', '+': '⠈⠚',
    ',': '⠂⠀', '-': '⠤⠀', '.': '⠲⠀', '/': '⠔⠀',
    ':': '⠈⠇', ';': '⠈⠍', '<': '⠈⠝', '=': '⠈⠕',
    '>': '⠈⠏', '?': '⠈⠟', '@': '⠈⠗', '[': '⠈⠎',
    '\\': '⠈⠞', ']': '⠈⠥', '^': '⠈⠧', '_': '⠈⠺',
    '`': '⠈⠭', '{': '⠈⠽', '|': '⠈⠵', '}': '⠈⠷',
    '~': '⠈⠾',
}

def convert_to_braille(text):
    braille_text = ""
    for char in text:
        braille_text += braille_map.get(char, char)

    return braille_text

# input text
text = """Abstract-Braille is one of the most important means
of written
communications between 
visually-impaired and sighted
people, so it gains the research interest. 
This paper describesa new technique for recognizing 
Braille characters in Arabic double sided Braille 
document. The main challenge resolved here is to 
build up a complete OBR system that is completely
invariant to scale of the scanned image starting 
from scanner passing image enhancement stages 
and followed by stages to detect parts of dots, 
then detecting the whole dot, and finally the 
Braille cell recognition. This technique can be
applied regardless the grade of the Braille 
document (Grade one or Grade two). Besides, 
the proposed stages up to the Braille cell 
recognition can be used in recognizing Braille
documents written in other languages too.

Abstract-Il braille è uno dei più importanti 
mezzi di comunicazione scritta tra ipovedenti 
e vedenti. Comunicazione scritta tra persone 
ipovedenti e vedenti. persone ipovedenti, per 
cui suscita l'interesse della ricerca. Questo 
documento descrive una nuova tecnica per 
il riconoscimento dei caratteri Braille in 
un documento arabo documenti Braille fronte/retro. 
La sfida principale risolta è quella di costruire 
un sistema OBR completo che sia completamente
invariante alla scala dell'immagine scansionata 
a partire dallo scanner passando per le fasi 
di miglioramento dell'immagine, seguite dalle 
fasi di di rilevamento di parti di punti, poi 
di rilevamento dell'intero punto e infine di
infine il riconoscimento della cella Braille. Questa tecnica può essere
Questa tecnica può essere applicata indipendentemente dal grado del documento Braille (grado uno o grado due).
grado o grado due). Inoltre, le fasi proposte fino al
riconoscimento delle celle Braille possono essere utilizzate per il riconoscimento di documenti Braille
documenti Braille scritti in altre lingue.

Già in epoca classica esisteva un uso "volgare" del latino, 
pervenutoci attraverso testi non letterari, graffiti, iscrizioni 
non ufficiali o testi letterari attenti a riprodurre la 
lingua parlata, come accade spesso nella commedia.[8] Accanto 
a questo, esisteva un latino "letterario", adottato dagli scrittori 
classici e legato alla lingua scritta, ma anche alla lingua 
parlata dai ceti socialmente più rilevanti e più colti.[8]
"""

# Con la caduta dell'Impero romano e la formazione dei regni 
# romano-barbarici, si verificò una sclerotizzazione del latino 
# scritto (che diventò lingua amministrativa e scolastica), mentre 
# il latino parlato si fuse sempre più intimamente con i dialetti 
# dei popoli latinizzati, dando vita alle lingue neolatine, 
# tra cui l'italiano.[9]

# Gli storici della lingua italiana etichettano le parlate che 
# si svilupparono in questo modo in Italia durante il Medioevo 
# come "volgari italiani", al plurale, e non ancora come "lingua italiana". 
# Le testimonianze disponibili mostrano infatti marcate differenze 
# tra le parlate delle diverse zone, mentre mancava un comune 
# modello volgare di riferimento.[senza fonte]

# Il primo documento tradizionalmente riconosciuto di uso di un 
# volgare italiano è un placito notarile, conservato nell'abbazia 
# di Montecassino, proveniente dal Principato di Capua e risalente 
# al 960: è il Placito cassinese (detto anche Placito di Capua o 
# "Placito capuano"), che in sostanza è una testimonianza giurata 
# di un abitante circa una lite sui confini di proprietà tra 
# il monastero benedettino di Capua afferente ai Benedettini 
# dell'abbazia di Montecassino e un piccolo feudo vicino, il 
# quale aveva ingiustamente occupato una parte del territorio 
# dell'abbazia: «Sao ko kelle terre per kelle fini que ki contene 
# trenta anni le possette parte Sancti Benedicti.» ("So [dichiaro] 
# che quelle terre nei confini qui contenuti (qui riportati) per 
# trent'anni sono state possedute dall'ordine benedettino").[7] 
# È soltanto una frase, che tuttavia per svariati motivi può 
# essere considerata ormai "volgare" e non più schiettamente 
# latina: i casi (salvo il genitivo Sancti Benedicti, che riprende 
# la dizione del latino ecclesiastico) sono scomparsi, sono presenti 
# la congiunzione ko ("che") e il dimostrativo kelle ("quelle"), 
# morfologicamente il verbo sao (dal latino sapio) è prossimo 
# alla forma italiana, ecc. Questo documento è seguito a brevissima 
# distanza da altri placiti provenienti dalla stessa area 
# geografico-linguistica, come il Placito di Sessa Aurunca 
# e il Placito di Teano.[senza fonte]
# """

# convert text to braille
text = convert_to_braille(text)
text += convert_to_braille(' '.join(braille_map.keys()))


# Create a new image
image = Image.new("RGB", (1024, 1376), "white")
draw = ImageDraw.Draw(image)

# Use a font that supports Braille characters
# font = ImageFont.truetype("arial.ttf", size=24)
font = ImageFont.truetype("DejaVuSans.ttf", size=24)


# Draw the Braille text on the image
draw.text((20, 20), text, font=font, fill="black")

# Save the image
image.save("test/braille.png")
