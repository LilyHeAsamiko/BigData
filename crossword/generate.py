import sys

from crossword import *


class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve_cannotwork(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        #return self.backtrack(dict())
        assignment = self.backtrack(dict())
        return assignment

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        C1
        """
        try:
            for var in self.domains:
                temp = self.domains[var]
                print('Bdomains',temp)
                for x in self.domains[var].copy():
                    if len(x) != var.length:
                        self.domains[var].remove(x)
                temp = {}
                print('Adomains',self.domains[var])
        except:
            raise NotImplementedError

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        
        Note that index by Overlap need to be checked as it hasn't passed consistence check
        """
        try: 
             if self.crossword.overlaps[x,y] is not None:
                 #if set(self.domains[x])-set([xx for (xx,yy) in self.crossword.overlaps[x,y] if xx in self.domains[x] and yy in self.domains[y]]) == set():
                 xx,yy = self.crossword.overlaps[x,y]
#                 print('vxdomaincopy',self.domains[x].copy(),'vydomaincopy',self.domains[y].copy())
                 for vx in self.domains[x].copy():
                     flag = False
                     for vy in self.domains[y].copy():
                         if xx >= len(vx) or yy >= len(vy):
                             continue
                         elif vy != vx and vx[xx] == vy[yy]:
                             flag = True
                             break
                     if flag == False:
                         self.domains[x].remove(vx)
                     else:
                         continue
                 return (True if flag == False else False)
             else:
                 return False
        except:
            raise NotImplementedError

    def revise_b(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        
        Note that index by Overlap need to be checked as it hasn't passed consistence check
        """
        try: 
             flag = False
             if self.crossword.overlaps[x,y] is not None:
                 #if set(self.domains[x])-set([xx for (xx,yy) in self.crossword.overlaps[x,y] if xx in self.domains[x] and yy in self.domains[y]]) == set():
                 xx,yy = self.crossword.overlaps[x,y]
#                 print('vxdomaincopy',self.domains[x].copy(),'vydomaincopy',self.domains[y].copy())
                 for vx in self.domains[x].copy():
                     for vy in self.domains[y].copy():
                         if xx >= len(vx) or yy >= len(vy):
                             continue
                         elif vx[xx] == vy[yy]:
                             flag = True
                     if flag == False:
                         self.domains[x].remove(vx)
                     else:
                         continue
                 return (True if flag == False else False)
             else:
                 [self.domains[x].remove(xitem) for xitem in self.domains[x].copy()] 
                 return True
        except:
            raise NotImplementedError


    def ac3_origin(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        
        Variable:
        self.variables(i,j,length)
        Domains:
        self.variables.cells
        Constrains:
        revise(xvariable,yvariable)
        arcs defaulted: All overlapped variables 
        """
        try:
            if arcs is None:
                arcs = set()
                for x in self.crossword.variables:
                    for y in self.crossword.neighbors(x):
                        arcs.add((x,y))
            print('arcsB',arcs)
            for (xx,yy) in arcs.copy():
                if self.domains[xx] != set() :
                    if self.revise(xx,yy):
                        self.revise(xx,yy)
                        [arcs.append(xx,zz) for zz in self.crossword.neighbors(xx)]
                elif self.domains[xx] == set():
                    return False
            return True
        except:            
            raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.
    
        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        
        Variable:
        self.variables(i,j,length)
        Domains:
        self.variables.cells
        Constrains:
        revise(xvariable,yvariable)
        arcs defaulted: All overlapped variables 
        """
        try:
            if arcs is None:
                arcs = set()
                for x in self.domains:
                    for y in self.domains:
                        if x != y and self.crossword.overlaps[x,y] is not None:
                            arcs.add((x,y))
            print('arcsB',arcs)
            queue = arcs
            for (xx,yy) in queue.copy():
                if len(queue) >0:                
                    queue.remove((xx,yy))
                    if self.revise(xx,yy):
                        if len(self.domains[xx]) == 0:    
                            return False
                        [queue.add((zz,xx)) for zz in self.crossword.neighbors(xx) if zz != yy]                           
                        print('queue',queue)
            return True
        except:
            raise NotImplementedError


    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to e
        crossword variable); return False otherwise.
        
        crossword.word stores the words
        """
        try:
            for v in self.domains: 
                if v not in assignment:
                    return False
            return True
        except:
            raise NotImplementedError

    def consistent_b(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        try:
            for word in asssignment:
                for v in self.crossword.words:
                    if v == word:
                        if assignment[v] in self.crossword.variables:
                            if v.length  == len(assignment[word]) and self.crossword.neighbors(v) is not None:
                                for neib in self.crossword.neighbors(v):
                                    vi, neibi = self.crossword.overlaps[v,neib]
                                    if v.cells[vi] == neib.cells[neibi]:
                                        return True
                            else:
                                return True
            return False
        except:
            raise NotImplementedError
            
    def consistent_ba(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        to be check:
        length
        assignment different values
        cross index
        """
        
        try:
            print('start consistence check with assignment', assignment)
            for vx in list(asssignment):
                print('vx',vx)
                print('varlen in assi',vx.length,'wordlen in assi',len(assignment[word]))
                if vx.length != len(assignment[word]):
                    return False
                else:
                    for vy in asssignment:
                        if vy != vx:
                            if assignment[vx] == assignment[vy]:
                                return False
                        print('checking assignment consistence')
                        if vx in self.crossword.neighbors(vx):
                            print('self neib')
                        elif vy in self.crossword.neighbors(vx):
                            vxi, vyi = self.crossword.overlaps[vx,vy]
                            if assignment[vx][vxi] != assignment[vy][vyi]:
                                return False
            print('consisence check TRUE')
            return True
        except:
            raise NotImplementedError
        
    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        to be check:
        length
        assignment different values
        cross index
        """        
        try:
            print('start consistence check with assignment', assignment)
            print('assignmenLen',len(list(assignment)))
            for vx in list(assignment):
                print('vx',vx)
                print('varlen in assi',vx.length,'wordlen in assi',len(list(assignment[vx])))
                if vx.length != len(list(assignment[vx])):
                    return False
                else:
                    for vy in list(assignment):
                        if vy != vx:
                            print('checking assignment consistence')
                            if assignment[vx] == assignment[vy]:
                                return False
                            elif vy in self.crossword.neighbors(vx):
                                if not self.crossword.overlaps[vx,vy]:
                                    return False
                                else:
                                    vxi, vyi = self.crossword.overlaps[vx,vy]
                                    if assignment[vx][vxi] != assignment[vy][vyi]:
                                        return False
            print('consistence check TRUE')
            return True
        except:
            raise NotImplementedError        
        
    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        try:
            listV = {} 
            for vd in self.domains[var]:
                lc = 0            
                if vd not in assignment.values():                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    for neib in self.crossword.neighbors(var):
                        if len(self.domains[neib])>0: 
                            if vd in self.domains[neib]: 
                                lc += 1
                    listV[vd]=lc
                    print('checking odv',vd,lc)
            return [k for k,v in sorted(listV.items(), key = lambda item:item[1])] 
        except:
            raise NotImplementedError

    def order_domain_values_b(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        try:
            listV = {} 
            for vd in self.domains[var]:
                lc = 0            
                if vd not in assignment.values():
                    print('checking odv')                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
                    for neib in self.crossword.neighbors(var):
                        if len(self.domains[neib])>0: 
                            vi, neibi = self.crossword.overlaps[var,neib]
                            print('stringcheck',self.domains[neib])
                            print('indexcheck',neibi)
                            for vd in self.domains[neib]: 
                                print('neibcheck',neib)
                                print('neibcellcheck',d)
                                if vd[vi] not in d or neibi >= neib.length:
                                    lc += 1
                    listV[vd]=lc
            return [k for k,v in sorted(listV.items(), key = lambda item:item[1])] 
        except:
            raise NotImplementedError


    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        try:
            minDN = 100000
            for var in self.domains:
                if var not in assignment:
                    if len(self.domains[var]) <= minDN:
                        theVar = var
                        minDN = len(self.domains[var])
                        print('unassignedD',self.domains[var])
            return theVar
        except:
            raise NotImplementedError

    def inference(self,var,assignment,method = 'MRV'):
        '''defaulted as MRV to maintain arcs with ac3 otherwise input the assignment 
           
           The Inference function runs the AC-3 algorithm as described. 
           Its output is all the inferences that can be made through enforcing arc-consistency. 
           Literally, these are the new assignments that can be deduced from the previous assignments and the structure of the constrain satisfaction problem.
        '''
        try: 
            if method == 'MRV':
                arcs = set()
                for neib in var.domains:
                    arcs.add((neib,var))
                print('arcsBI',arcs)
                flagC = len(arcs)
                while arcs:
                    ac3(arcs)
                    #works or not
                    if len(arcs) == flagC and len(arcs)>2:
                        return False
                    #if not failure assign 
                    elif len(arcs) <= 2:
                        if consistent(assignment):
                            return arcs
                        else:
                            arcs.remove(var)
                    flagC = len(arcs)
            else:
                return assignment
        except:
            raise NotImplementedError

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        
        Inference with MRV
        """
        try:
            if self.assignment_complete(assignment):
                print('complete!')
                return assignment
            var = self.select_unassigned_variable(assignment)
            print('unassignedvar',var)
            print('assignmentInit',assignment)
            print('unassignedlist',self.order_domain_values(var, assignment))
            for value in list(self.order_domain_values(var, assignment)):
                print('1111111111111111111111111111111111')
                #(print("TRUE") if self.consistent(assignment) else print("FALSE"))
                if not self.consistent(assignment):
                    print("FALSE")
                print('c0 assignment',self.consistent(assignment))
                assignment[var] = value
                print('assignment before consistent check',assignment)
                print('c1 assignment',self.consistent(assignment))
                if self.consistent(assignment):
                    self.domains[var] = {value}
#                    inferences = inference(assignment,method = 'MVR')
#                    if inferences is not False:
#                         for infers in inferences:
#                              assignment[infers] = inferences[infers].cells
                    self.ac3(arcs = set([(x,var) for x in self.crossword.neighbors(var) if x != var and self.crossword.overlaps[x,var] is not None]))
                    print('ad',self.domains)
                    print('AI',assignment)
                    result = self.backtrack(assignment)
                    print('result',result)
                    if result is not None:
                        return result
                assignment.pop(var)
            return None
        except:
            raise NotImplementedError

def main():
    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None
    
    # Generate crossword
    crossword = Crossword(structure, words)
    #    print('crswd',crossword.structure,crossword.words,crossword.variables)
    creator = CrosswordCreator(crossword)
#    print(crossword.variables)
    assignment = creator.solve()
    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
