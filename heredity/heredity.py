import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people) 
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
#            print(have_trait,'have trait fails_evidence')
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):
                
                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
 #               print('1:',one_gene,'\n','2:',two_genes,'\n','have trait:',have_trait,'\n',people,'\n','joint probability:',p)
                update(probabilities, one_gene, two_genes, have_trait, p)
 #               print(probabilities,'\n...............')
    # Ensure probabilities sum to 1
 #   print('before normalization\n',probabilities,'\n------------------')
    normalize(probabilities)
 #   print('after normalization\n',probabilities,'\n------------------')

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability_before(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
          not from father * from mother + not from mother * from father
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
        
    Only consider with parents listed or no parents listed instead of with one parent listed
    father/mother with one gene might pass the gene to child with 0.5 possibility(not based on having trait or not) in addition to the mutation case
    """
    try:
        probabilityJ = 1
        for person in people:
            p = 0
            if people[person]['mother'] is None and people[person]['father'] is None:
                if person in one_gene:
                    p += PROBS["gene"][1]
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][1][True]
                    else:
                        p*= PROBS["trait"][1][False]
                elif person in two_genes:
                    p += PROBS["gene"][2]
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][2][True]
                    else:
                        p*= PROBS["trait"][2][False]
                elif person not in one_gene and person not in two_genes:
                    p += PROBS["gene"][0]
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][0][True]
                    else:
                        p*= PROBS["trait"][0][False]
            else:
                father  = people[person]['father']
                mother  = people[person]['mother']
                if person in one_gene: 
                # not from mother * from father 
                    if mother not in one_gene and mother not in two_genes:
                        pm = 1-PROBS["mutation"]
                    elif mother in one_gene:
                        pm = 0.5
                    elif mother in two_genes:
                        pm = PROBS["mutation"]
                    if father not in one_gene and father not in two_genes:
                        pf = PROBS["mutation"]
                    elif father in one_gene:
                        pf = 0.5
                    elif father in two_genes:
                        pf = 1-PROBS["mutation"]
                    p += pm*pf    
                #+ not from father * from mother
                    if mother not in one_gene and mother not in two_genes:
                        pm = PROBS["mutation"]
                    elif mother in one_gene:
                        pm = 0.5
                    elif mother in two_genes:
                        pm = 1-PROBS["mutation"]
                    if father not in one_gene and father not in two_genes:
                        pf = 1-PROBS["mutation"]
                    elif father in one_gene:
                        pf = 0.5
                    elif father in two_genes:
                        pf = PROBS["mutation"]
                    p += pm*pf    
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][1][True]
                    else:
                        p *=PROBS["trait"][1][False]
                elif person in two_genes:
                    # from mother * from father 
                    if mother not in one_gene and mother not in two_genes:
                        pm = PROBS["mutation"]
                    elif mother in one_gene:
                        pm = 0.5
                    elif mother in two_genes:
                        pm = 1-PROBS["mutation"]
                    if father not in one_gene and father not in two_genes:
                        pf = PROBS["mutation"]
                    elif father in one_gene:
                        pf = 0.5
                    elif father in two_genes:
                        pf = 1-PROBS["mutation"]
                    p += pm*pf    
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][2][True]
                    else:
                        p *=PROBS["trait"][2][False]
                elif person not in one_gene and person not in two_genes:
                    # not from mother * not from father 
                    if mother not in one_gene and mother not in two_genes:
                        pm = 1-PROBS["mutation"]
                    elif mother in one_gene:
                        pm = 0.5
                    elif mother in two_genes:
                        pm = PROBS["mutation"]
                    if father not in one_gene and father not in two_genes:
                        pf = 1-PROBS["mutation"]
                    elif father in one_gene:
                        pf = 0.5
                    elif father in two_genes:
                        pf = PROBS["mutation"]
                    p += pm*pf  
                    if people[person]['trait'] is True:
                        p *= PROBS["trait"][0][True]
                    else:
                        p *=PROBS["trait"][0][False]
            print(person,p)
            probabilityJ *= p
        return probabilityJ            
    except:
        raise NotImplementedError

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
          not from father * from mother + not from mother * from father
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
        
    Only consider with parents listed or no parents listed instead of with one parent listed
    father/mother with one gene might pass the gene to child with 0.5 possibility(not based on having trait or not) in addition to the mutation case
    """
    try:
        probabilityJ = 1
        for person in people:
            p = 0
            geneN = (1 if person in one_gene else 2 if person in two_genes else 0)
            traitP = (True if person in have_trait else False)
            if people[person]['mother'] is None and people[person]['father'] is None:
                p = PROBS["gene"][geneN] * PROBS["trait"][geneN][traitP]
            else:
                father  = people[person]['father']
                mother  = people[person]['mother']                
                if person in one_gene: 
                # not from mother * from father 
                    pm = (PROBS["mutation"] if mother in two_genes else 0.5 if mother in one_gene else (1-PROBS["mutation"]))
                    pf = ((1-PROBS["mutation"]) if father in two_genes else 0.5 if father in one_gene else PROBS["mutation"])
                    p += pm*pf    
                #+ not from father * from mother
                    pm = ((1-PROBS["mutation"]) if mother in two_genes else 0.5 if mother in one_gene else PROBS["mutation"])
                    pf = (PROBS["mutation"] if father in two_genes else 0.5 if father in one_gene else (1-PROBS["mutation"]))
                    p += pm*pf    
                elif person in two_genes:
                    # from mother * from father 
                    pm = ((1-PROBS["mutation"]) if mother in two_genes else 0.5 if mother in one_gene else PROBS["mutation"])
                    pf = ((1-PROBS["mutation"]) if father in two_genes else 0.5 if father in one_gene else PROBS["mutation"])
                    p += pm*pf       
                elif person not in one_gene and person not in two_genes:
                    # not from mother * not from father 
                    pm = (PROBS["mutation"] if mother in two_genes else 0.5 if mother in one_gene else (1-PROBS["mutation"]))
                    pf = (PROBS["mutation"] if father in two_genes else 0.5 if father in one_gene else (1-PROBS["mutation"]))
                    p += pm*pf       
                p *=PROBS["trait"][geneN][traitP]
#            print(person,p)
            probabilityJ *= p
        return probabilityJ            
    except:
        raise NotImplementedError

def update_origin(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    try:
        for person in probabilities:
            if person in one_gene:
                probabilities[person]["gene"][1] += p
            elif person in two_genes:
                probabilities[person]["gene"][2] += p
            elif person not in one_gene and person not in two_genes:
                probabilities[person]["gene"][0] += p
            if person in have_trait:
                probabilities[person]["trait"][True] += p
            elif person not in have_trait:
                probabilities[person]["trait"][False] += p    
        probabilities.update()
    except:
        raise NotImplementedError
        
def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    try:
        for person in probabilities:
            geneN = (1 if person in one_gene else 2 if person in two_genes else 0)
            traitP = (True if person in have_trait else False)
            probabilities[person]["gene"][geneN] += p
            probabilities[person]["trait"][traitP] += p
        probabilities.update()
    except:
        raise NotImplementedError

def normalize_origin(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    try:
        for person in probabilities:
            print('before:\n',person,probabilities[person]["trait"][True],probabilities[person]["trait"][False],sum(probabilities[person]["trait"].values()),probabilities[person]["gene"][2],probabilities[person]["gene"][1],probabilities[person]["gene"][0],sum(probabilities[person]["gene"].values()))
            probabilities[person]["trait"][True] /= sum(probabilities[person]["trait"].values())
            probabilities[person]["trait"][False] /= sum(probabilities[person]["trait"].values())
            probabilities[person]["gene"][2] /= sum(probabilities[person]["gene"].values())
            probabilities[person]["gene"][1] /= sum(probabilities[person]["gene"].values())
            probabilities[person]["gene"][0] /= sum(probabilities[person]["gene"].values())
            print('after:\n',person,probabilities[person]["trait"][True],probabilities[person]["trait"][False],sum(probabilities[person]["trait"].values()),probabilities[person]["gene"][2],probabilities[person]["gene"][1],probabilities[person]["gene"][0],sum(probabilities[person]["gene"].values()))
        probabilities.update()
        print('after:\n',probabilities)
    except:
        raise NotImplementedError

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    try:
        for person in probabilities:
            sumTrait = sum(probabilities[person]["trait"].values())
            sumGene = sum(probabilities[person]["gene"].values())
            probabilities[person]["trait"][True] /= sumTrait
            probabilities[person]["trait"][False] /= sumTrait
            probabilities[person]["gene"][2] /= sumGene
            probabilities[person]["gene"][1] /= sumGene
            probabilities[person]["gene"][0] /= sumGene
        probabilities.update()
    except:
        raise NotImplementedError


if __name__ == "__main__":
    main()
