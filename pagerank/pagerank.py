import os
import random
import re
import sys
import numpy as np

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    #somehow the result of sampling doesn't seem the same as the example, so I try consider using median of 100 of the method
    RK = []
    for i in range(100):
        RK.append(list(sample_pagerank(corpus, DAMPING, SAMPLES).values()))
    j = 0
    for i in sorted(ranks):
        ranks[i] = np.median(RK,axis = 0)[j]
        j += 1
    print(f"PageRank Results from SamplingAVG (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    #maxium iterstep set as 1000 with convergency of sse (usually converges fast around 10 steps)
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    try:
        PR = {}
        pages = corpus.keys()
        nd = len(corpus[page])
        for p in set(pages):
            PR[p] = 1/(len(pages))
#        print(PR)
        if corpus[page] != set():
            for p in set(pages):
                if p not in corpus[page]:
                    PR[p] *= (1-damping_factor)
                else:
                    PR[p] = (1-damping_factor)*PR[p] + damping_factor/nd
        return PR
    except:
        raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    try:
        PR = {}
        pages = corpus.keys()
        for p in pages:
            PR[p] = 0
        page = random.sample(pages,1)
        P = transition_model(corpus, page[0], damping_factor)
        PR[page[0]] += 1 
        for i in range(1,n):
            page = random.choices(list(P.keys()),weights=list(P.values()),k=1)
            PR[page[0]] += 1
            P = transition_model(corpus, page[0], damping_factor)
        for p in pages:
            PR[p] /=  n
        return PR
    except:
        raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    
    Note here: iter_steps maximum is 100 and convergency set as 0.001 with sse
    """
    try:
        def PR_iteration(pr, corpus, damping_factor):
            newPR = {}
            for p in pr.keys():
                newPR[p] = 0
            for p in set(corpus.keys()):
                for q in set(corpus.keys()):
                    if corpus[p] != set():
                        if q in corpus[p]:
                            newPR[q] += damping_factor*pr[p]/len(corpus[p])
                newPR[p] += (1-damping_factor)/len(corpus.keys())                
            return newPR
        PR = {}
        pages = corpus.keys()
        #initiate
        iters = 1
        for p in set(pages):
            PR[p] = 1/(len(pages))
#        print(PR,iters)
        formerPR = PR
        #iterate until converges         
        newPR = PR_iteration(PR, corpus, damping_factor)
        diff = sum([abs(newPR[d]-formerPR[d]) for d in newPR.keys()])**0.5
        checkPR = []
        checkDiff = []
        while diff > 0.001 and iters <1000:
            PR = newPR
            checkPR.append(PR)
            checkDiff.append(diff)            
#            print(PR,diff,iters)
            newPR = PR_iteration(PR, corpus, damping_factor)
#            diff = sum([abs(newPR[d]-PR[d]) for d in newPR.keys()])
            diff = sum([(newPR[d]-PR[d])**2 for d in newPR.keys()])**0.5
            iters += 1
        #return PR,checkPR,checkDiff
        return PR
    except:
        raise NotImplementedError



if __name__ == "__main__":
    main()
