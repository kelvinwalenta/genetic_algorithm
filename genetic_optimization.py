# Kelvin Walenta 11771895

import pyscf
import numpy as np
import random
import copy
import matplotlib.pyplot as plt
import os
import sys
import time



def get_coordinates() -> dict:
    
    '''
    This function takes the z-matrix values and writes them into a dictionary 
    for easier modification of certain degrees of freedom. The keys are the species 
    of the atom combined with the row index of the input file.
    '''

    '''
    coordinates = {
                'C1': ['C'], 'C2': ['C',1,1.4], 'C3': ['C',2,1.4,1,120], 'C4': ['C',3,1.4,2,120,1,0],
                'C5': ['C',4,1.4,3,120,2,0], 'C6': ['C',5,1.4,4,120,3,0], 'C7': ['C',6,1.34,5,120,4,180],
                'C8': ['C',1,1.34,2,120,3,180], 'C9': ['C',8,1.34,1,120,2,180], 'C10': ['C',7,1.34,6,120,5,180], 
                'O1': ['O',7,1.24,6,120,5,0], 'H1': ['H',11,0.89,7,120,6,180], 'O2': ['O',5,1.24,4,120,3,180], 
                'H2': ['H',13,0.89,5,120,6,180], 'O3': ['O',3,1.24,4,120,5,180], 'H3': ['H',15,0.89,3,120,4,180],
                'O4': ['O',8,1.24,1,120,6,180], 'O5': ['O',9,1.24,8,120,1,180], 'H4': ['H',17,0.89,8,120,9,180],
                'H5': ['H',18,0.89,9,120,8,180], 'X1': ['H',10,0.99,7,120,6,180], 'X2': ['F',2,1.39,3,109.471,4,180],
                'X3': ['F',4,1.39,3,120,2,180]
                }

    '''
    coordinates = {
                'C1': ['C'], 'C2': ['C',1,1.4], 'C3': ['C',2,1.4,1,120], 'C4': ['C',3,1.4,2,120,1,0],
                'C5': ['C',4,1.4,3,120,2,0], 'C6': ['C',5,1.4,4,120,3,0], 'C7': ['C',6,1.34,5,120,4,180],
                'C8': ['C',1,1.34,2,120,3,180], 'C9': ['C',8,1.34,1,120,2,180], 'C10': ['C',7,1.34,6,120,5,180], 
                'O1': ['O',7,1.24,6,120,5,0], 'H1': ['H',11,0.89,7,120,6,180], 'O2': ['O',5,1.24,4,120,3,180], 
                'H2': ['H',13,0.89,5,120,6,180], 'O3': ['O',3,1.24,4,120,5,180], 'H3': ['H',15,0.89,3,120,4,180],
                'O4': ['O',8,1.24,1,120,6,180], 'O5': ['O',9,1.24,8,120,1,180], 'H4': ['H',17,0.89,8,120,9,180],
                'H5': ['H',18,0.89,9,120,8,180], 'X1': ['F',2,1.39,3,109.471,4,180], 'X2': ['Cl',4,1.74,5,109.471,6,180],
                'X3': ['F',10,1.39,9,109.471,8,180]
                }           
    
    return coordinates


def dict_to_input(coordinates: dict) -> str:

    '''
    This function takes the dictionary 'coordinates' and transforms it into a 
    single string as input for the 'pySCF' module.
    '''

    input_str = ''

    for key in coordinates:

        input_str += ' '.join(map(str,coordinates[key])) + '; '

    return input_str


def print_genom(coordinates: dict) -> None:
    
    dofs = ['H1','H2','H3','H4','H5']

    genom = []

    for dof in dofs:

        genom.append(coordinates[dof][-1])
    
    print(genom)

    return None


def plot_energy(E_min: np.ndarray, E_mean: np.ndarray, dirname: str, n: int, mut_prob: float, sel_prob: float, pairing: str, run: int) -> None:

    """
    This function plots the energy evolution of the geometry optimization.
    """

    path = dirname + '/energy_{}.png'.format(run)

    _, ax = plt.subplots()
    ax.ticklabel_format(useOffset=False)
    # ax.ticklabel_format(useOffset=False, style='plain')
  
    ax.plot(E_min,'--k', marker = 'o', ms = 6, mec = 'k', mfc = 'blue',label=r'$E_{min}$')
    ax.plot(E_mean,'--k', marker = 'o', ms = 6, mec = 'k', mfc = 'orange',label=r'$\bar{E}$')

    ax.set_xlabel(r'n', fontsize=12)
    ax.set_ylabel(r'E[Eh]', fontsize=12)
    if pairing == 'single':
        ax.set_title(r'Population size: {0}, Mutation: {1}%, Single Mating'.format(n, mut_prob*100), fontsize=12)
    else:
        ax.set_title(r'Population size: {0}, Mutation: {1}%, Uniform Mating'.format(n, mut_prob*100), fontsize=12)
    ax.legend(loc='upper right', prop={'size': 12})
    # plt.xticks(np.arange(0,50,1))
    plt.grid()
    plt.savefig(path,dpi=200, bbox_inches = "tight")
    plt.close()
    #plt.show()


    return None 

     
def brute_force() -> None:
    """
    This function brute forces every possible structure and dtermines the corresponding energy.
    """

    temp = get_coordinates()
    angles = np.array([0,180])

    for dof1 in angles:
        temp['H1'][-1] = dof1
        for dof2 in angles:
            temp['H2'][-1] = dof2
            for dof3 in angles:
                temp['H3'][-1] = dof3
                for dof4 in angles:
                    temp['H4'][-1] = dof4
                    for dof5 in angles:
                        temp['H5'][-1] = dof5

                        input_str = dict_to_input(temp)
                        mol = pyscf.M(

                            atom = input_str, 
                            basis = 'sto-3g',
                            symmetry = True,
                        )

                        RHF_calc = mol.HF() # Hartree-Fock approach
                        energy = RHF_calc.kernel() # Energy  
                        print_genom(temp)
                                      
    return None
    
    
class population:


    def __init__(self,coordinates: dict, n: int, max_it: int) -> None:

        self.coordinates = coordinates # dictionary that represents the z-matrix
        self.n = n # population size
        self.max_it = max_it # maximum number of iterations
        self.dofs = ['H1','H2','H3','H4','H5'] # atoms with degrees of freedom
        self.pop = [] # initialize population
        self.E_opt = 1e9 # running variable for the best achieved energy, high value for initialization
        self.struc_opt = [] # running variable that stores the dictionary that represents the best structure
        self.history = {} # dictionary that will contain the 'genome' of every structure visited and its corresponding energy
        # self.angles = np.array([0,180])
        self.angles = np.arange(0,360,60)
        self.conv_counter = 0


    def initialize(self) -> list:

        '''
        This function takes the dictionary 'coordinates', changes the dihedral angles of the 
        H atoms to a randomly chosen single point from the array 'angles' and appends it to a list 'pop'.
        This is repeated 'n' times where 'n' is the size of the population. 
        '''


        for _ in range(self.n):
  
            temp = copy.deepcopy(self.coordinates)

            for dof in self.dofs:

                temp[dof][-1] = random.choice(self.angles) # [-1] represents the dihedral angle (last element in each row of the z matrix)
            
            self.pop.append(temp)

        return self.pop

        
    def evaluate(self) -> np.ndarray:

        '''
        This function calculates the energy of each individual of the population using the 'pyscf' 
        package. As the 'strength' of all previous individuals is saved in 'history', it is first always checked whether 
        the current individual already existed to avoid calculating the energy again.
        In addidtion, it is checked whether or not a new global 'best' structure was found. 
        If so, it is saved as 'struc_opt'.
        '''

        E = np.zeros(self.n) # initialize energies
       
        for i in range(self.n):

            # representation of a individual, eg 18012060180240
            genes_key = int(f"{self.pop[i]['H1'][-1]}{self.pop[i]['H2'][-1]}{self.pop[i]['H3'][-1]}{self.pop[i]['H4'][-1]}{self.pop[i]['H5'][-1]}")
           

            if genes_key in self.history: # check if it already existed

                E[i] = self.history[genes_key]
                print('converged SCF energy = %f (from history)' %E[i])

            else:

                input = dict_to_input(self.pop[i])
                mol = pyscf.M(

                    atom = input, 
                    basis = 'sto-3g',
                    symmetry = True,
                    )

                RHF_calc = mol.HF() # Hartree-Fock approach
                E[i] = RHF_calc.kernel() # Energy  
                self.history[genes_key] = E[i]
                

        index  = np.argmin(E) # index of best energy

        if E[index] < self.E_opt: # save best configuration

            self.E_opt = E[index]
            self.struc_opt = self.pop[index]
            self.conv_counter = 0

        return E


    def select(self,E: np.ndarray, sel_prob: int) -> list: 

        '''
        This function selects 'two' individuals randomly for mating distributed according to their energy.
        '''


        # VARIANT A FOR DECISION

        # temp = np.argsort(-E) # indices that would sort the array
        # rank = np.argsort(temp) + 1 # ranking of each individual in the same order as E values
        # rank = rank/np.sum(rank)

        # VARIANT B FOR DECISION:

        temp = np.exp(-sel_prob*(E-np.max(E))) # factor sel_prob determines how sharp we choose the parents
        # temp = np.exp(sel_prob*(E-np.max(E))) # choose the weakest for mating
        rank = temp/np.sum(temp)
        
        '''
        Note:   With random.choices() individuals of a population are chosen in such a way, that the same individual
                can be chosen more than once. As this is not in accordance with the genetic algorithm, 
                np.random.choice() is used with the keyword 'replace=False'.
        '''
        # parents = random.choices(population=self.pop, weights = rank, k = 2) 

        parents =  []
        proposal = np.random.choice(self.n,2,replace=False,p=rank) # better as each individual can only be chosen once

        parents.append(self.pop[proposal[0]])
        parents.append(self.pop[proposal[1]])

        return parents


    def mate_single(self, parents: list) -> list: 

        ''' 
        This function takes the 'parents' from the function select and pairs them.
        The pairing is done by a single crossover where at some random point the genom is split.
        '''

        next_gen = []

        offspring_a = copy.deepcopy(parents[0]) # initialize offspring
        offspring_b = copy.deepcopy(parents[0])
            
        number_dofs = len(self.dofs)
        split_pos = np.random.randint(1,number_dofs - 1) # choose random position where genom is split

        for dof in self.dofs[:split_pos]:

            offspring_a[dof][-1] = parents[1][dof][-1] 

        next_gen.append(offspring_a)

        for dof in self.dofs[split_pos:]:

            offspring_b[dof][-1] = parents[1][dof][-1] 

        next_gen.append(offspring_b)

        return next_gen 


    def mate_uniform(self, parents: list) -> list: 

        ''' 
        This function takes the 'best' individuals from the function select and pairs them with their 
        neighbor in the list. The pairing is done by randomly chosing each gene from one of the two
        partners. 
        '''

        next_gen = []

        offspring_a = copy.deepcopy(parents[0]) # initialize offspring
        offspring_b = copy.deepcopy(parents[0])
        """
        for i in range(len(alpha_pop)):
            
            temp = alpha_pop[i]

            for dof in self.dofs:

                temp[dof][-1] = random.choice([alpha_pop[i-1][dof][-1],alpha_pop[i][dof][-1]])
            
            next_gen.append(temp)
        """

        for dof in self.dofs:

            offspring_a[dof][-1] = random.choice([parents[0][dof][-1],parents[1][dof][-1]])
            offspring_b[dof][-1] = random.choice([parents[0][dof][-1],parents[1][dof][-1]])
            
        next_gen.append(offspring_a)
        next_gen.append(offspring_b)

        return next_gen 

 
    def mutate_inversion(self,individual: dict, mutate_prop: float) -> list:

        '''
        This function takes an individual, chooses two random numbers that define a section of the genome and inverts the order 
        of the genes in this section. 
        '''

        if np.random.rand() < mutate_prop:

            [a,b] = sorted(np.random.choice(range(len(self.dofs)), size=2, replace=False))

            temp = copy.deepcopy(individual)


            for i in range(b-a+1):

                gene_a = self.dofs[a+i]
                gene_b = self.dofs[b-i]

                individual[gene_a][-1] = temp[gene_b][-1]

            # print_genom(individual)

        return individual


    def mutate(self,individual: dict, mutate_prop: float) -> list: 

        '''
        This function takes one individual as input and performs a random mutation with a probability 
        defined by mutate_prob.
        '''
        
        if np.random.rand() < mutate_prop:

            gene = random.choice(self.dofs) # choose gene to mutate
            individual[gene][-1] = random.choice(self.angles)

        return individual


    def evolve(self,mut_prob: float, sel_prob: float, conv: int, mating: str) -> list: 

        '''
        This function performs the evolution of the population. In a loop, the population is evaluated, individuals
        for mating are selected and mutation is performed. For each generation, the mean as well as the minimu energy is evaluated.
        This is repeated until convergence is reached.
        '''

        counter = 0 # initialization

        E_mean = []
        E_min = []

        while counter < self.max_it: # start evolution

            print('-------------------------------------------')
            print('Generation %i :' %counter)

            E = self.evaluate() # evaluate energy of individuals in population

            E_mean.append(np.mean(E))
            E_min.append(np.min(E))

            next_gen = []

            print('Mean energy of population: %f ' % E_mean[-1])
            print('Minimum energy of population: %f ' % E_min[-1])

            # Mating:

            for _ in range(int(self.n/2 - 1)): # create n-2 children for next generation, instead of two, parameter k???

                parents = self.select(E,sel_prob)

                if mating == 'single':
    
                    offspring_a, offspring_b = self.mate_single(parents)

                    new_a = self.mutate_inversion(offspring_a, mut_prob)
                    new_b = self.mutate_inversion(offspring_b, mut_prob)

                    # new_a = self.mutate(offspring_a, mut_prob)
                    # new_b = self.mutate(offspring_b, mut_prob)

                    next_gen.append(new_a)
                    next_gen.append(new_b)
                else:
                    offspring_a, offspring_b = self.mate_uniform(parents)

                    new_a = self.mutate_inversion(offspring_a, mut_prob)
                    new_b = self.mutate_inversion(offspring_b, mut_prob)

                    # new_a = self.mutate(offspring_a, mut_prob)
                    # new_b = self.mutate(offspring_b, mut_prob)

                    next_gen.append(new_a)
                    next_gen.append(new_b)
            
            # Add the two (or k?) best individuals from the parents generation to the next generation:

            ranked = np.argsort(E)

            next_gen.append(self.pop[ranked[0]])
            next_gen.append(self.pop[ranked[1]])

            print_genom(self.pop[ranked[0]])

            self.pop = next_gen

            counter += 1
            self.conv_counter += 1

            if self.conv_counter == conv:
                break


        return E_mean, E_min


def main():
    # Input parameter list
    n_list = [12]                   # for convenience, define it as even number
    max_it = 50                     # max number of iterations 
    sel_prob_list = [0,30,3000]     # higher number results in sharper selection if exponential distribution is used, otherwise irrelevant
    coordinates = get_coordinates() # transform z matrix into dictionary which represents individual
    conv = 15                       # convergence criterion; if best value stays the same conv times exit the loop
    pairing = 'uniform'             # Parent mating, either single or uniform
    mut_prob = [0.5]                # Higher number leads to longer search, but less likely to get stuck in local minima
    n_runs = 1

    # One multiloop for all relevant parameters

    for sel_prob in sel_prob_list:

        for n in n_list:

            if n % 2  != 0:
                raise ValueError('Population size needs to be an even number!')
            #for i, pm in enumerate(mut_prob):
            for pm in mut_prob:

                dirname = 'n_' + str(n)  + '_sel_' + str(sel_prob) + '_mut_' + str(pm) + '_' + pairing 
                    
                try:                        
                    os.makedirs(dirname)  
                               
                except OSError:
                    pass  # already exists

                for run in range(1,n_runs+1):

                    sys.stdout = open(dirname + '/' + 'output' +  '_' + str(run) + '.txt', "w")

                    p = population(coordinates,n,max_it)
                    p.initialize() # initialize population

                    t_0 = time.time()
                    E_mean, E_min = p.evolve(pm,sel_prob,conv,pairing)
                    t_1 = time.time()

                    print('Computation time: %.1f s' %(t_1-t_0))
                    print('Genome of the best individual:')
                    print_genom(p.struc_opt)
                    print('Minimum energy reached: %f ' % p.E_opt)
                    plot_energy(E_min, E_mean, dirname, n, pm, sel_prob, pairing, run)
                
                    del p

                    sys.stdout.close()

    return None


if __name__ == "__main__":
    
    main()


