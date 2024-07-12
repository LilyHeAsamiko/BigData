import nltk
import sys
from nltk.tree import Tree, ParentedTree
nltk.download('punkt')

TERMINALS = """
Adj -> "country" | "dreadful" | "enigmatical" | "little" | "moist" | "red"
Adv -> "down" | "here" | "never"
Conj -> "and" | "until"
Det -> "a" | "an" | "his" | "my" | "the"
N -> "armchair" | "companion" | "day" | "door" | "hand" | "he" | "himself"
N -> "holmes" | "home" | "i" | "mess" | "paint" | "palm" | "pipe" | "she"
N -> "smile" | "thursday" | "walk" | "we" | "word" | "city" | "world"
P -> "at" | "before" | "in" | "of" | "on" | "to"
V -> "arrived" | "came" | "chuckled" | "had" | "lit" | "said" | "sat" | "saw"
V -> "smiled" | "tell" | "were" | "hello"
"""

NONTERMINALS = """
S -> NP VP | S Conj S | VP NP | S P S | S NP | S P NP
NP  -> N | Det AA N | Det N | NP Adv V | AA N  | Det N AA | P NP
AA -> Adj | Adj AA | Adv
VP -> V| V P NP | Adv V| V P| V AA
"""

grammar = nltk.CFG.fromstring(NONTERMINALS + TERMINALS)
parser = nltk.ChartParser(grammar)


def main():

    # If filename specified, read sentence from file
    if len(sys.argv) == 2:
        with open(sys.argv[1]) as f:
            s = f.read()

    # Otherwise, get sentence as input
    else:
        s = input("Sentence: ")

    # Convert input into list of words
    s = preprocess(s)

    # Attempt to parse sentence
    try:
        trees = list(parser.parse(s))
    except ValueError as e:
        print(e)
        return
    if not trees:
        print("Could not parse sentence.")
        return

    # Print each tree with noun phrase chunks
    for tree in trees:
        tree.pretty_print()

        print("Noun Phrase Chunks")
        for np in np_chunk(tree):
            print(" ".join(np.flatten()))
            #print(" ".join(np[:]))


def preprocess(sentence):
    """
    Convert `sentence` to a list of its words.
    Pre-process sentence by converting all characters to lowercase
    and removing any word that does not contain at least one alphabetic
    character.
    ASCII 0-9<A-Z<a-z
    """
    try:
        tokens = nltk.word_tokenize(sentence)
        print('before',tokens)
        for tki in range(len(tokens)):
            if False not in [ord(s)<ord('A') or ord(s)>ord('z')  for s in tokens[tki]]:
                tokens.pop(tki)
            else:
                tokens[tki] = tokens[tki].lower()
        print('after',tokens)
        return tokens
    except:    
        raise NotImplementedError


def np_chunk(tree):
    """
    Return a list of all noun phrase chunks in the sentence tree.
    A noun phrase chunk is defined as any subtree of the sentence
    whose label is "NP" that does not itself contain any other
    noun phrases as subtrees.
    """
    npList = []
    try:
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                npList.append(subtree)
        return npList
    except ValueError:
        print("No parse tree possible.")


if __name__ == "__main__":
    main()
