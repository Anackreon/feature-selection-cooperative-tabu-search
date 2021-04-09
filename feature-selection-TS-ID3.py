import sys
import random
import time
import pandas as pd
import numpy as np
from queue import deque
import re


"""
Configuration parameters
"""
FEATURE_SET_SIZE = 100      # Size of initial feature set
SUBSET_SIZE = 30            # Size of the selected feature set
TS_MOVES = 20               # Amount of putative solutions
LOOP_TRIES = 500            # How many potential moves to try in one move attempt
MAX_MOVES = 2000            # Maximum amoun of iterations the TS will have
TABU_LIST_SIZE = 250        # The number of memorized tabooed solutions
ENABLE_IG_CACHE = True      # Cache the results of information gain calculation; disabling impacts performance
ACCEPT_THRESHOLD = 0.2935   # Threshold value used in the acceptance criteria; set to 0 to disable
ENABLE_EXEC_LOG = True      # Enable detailed logs describing the algoritm execution



class DataLoader:
    """
    Provides functionality for loading datasets from file
    """
    def load(dataFile):
        """
        Loads data from a file in predefined format and returns it in a structured data set
        """
        #Import the dataset and define the feature as well as the target datasets / columns
        dataset = pd.read_table(dataFile, delim_whitespace=True, names=('A', 'B'))
        dataset["A"] = dataset["A"].apply(list)
        #Initialize the list of feature names
        col_names = []
        for i in list(range(FEATURE_SET_SIZE)):
            col_names.append('feature_'+ str(i))
        #Apply feture names to the dataset
        dataset = pd.concat([dataset['A'].apply(pd.Series, index=col_names), dataset["B"]], axis=1)
        return dataset
    
    def getFeatures(dataset):
        """
        Returns the complete list of features of a given data set
        """
        return dataset.columns[:-1]

    def sortFeatureSet(features):
        """
        Sorts in a human expected way a feature set consisting from alpha-numeric values and returns it as a list
        """
        featList = list(features)
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
        return sorted(featList, key = alphanum_key)



class ID3:
    """
    Implements methods for calculating a dataset entropy and information gain of a selected feature set
    """
    def __init__(self):
        """
        Initialization method
        """
        if ENABLE_IG_CACHE:
            self.solutionsIgCache = dict()
            print("ID3 initialized with cache...")
        else:
            print("ID3 initialized...")
    
    def entropy(self, target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements,counts = np.unique(target_col,return_counts = True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    def infoGain(self, data, split_attribute_name, target_name="B"):
        """
        Calculate the information gain of a dataset. This function takes three parameters:
        1. data = The dataset for whose feature the IG should be calculated
        2. split_attribute_name = the name of the feature for which the information gain should be calculated
        3. target_name = the name of the target feature. The default for this example is "class"
        """
        #If cache enabled, first try find calculated information gain
        if ENABLE_IG_CACHE and split_attribute_name in self.solutionsIgCache.keys():
            return self.solutionsIgCache.get(split_attribute_name)

        #Calculate the entropy of the total dataset
        totalEntropy = self.entropy(data[target_name])
        
        ##Calculate the entropy of the dataset
        #Calculate the values and the corresponding counts for the split attribute 
        vals,counts = np.unique(data[split_attribute_name],return_counts=True)
        #Calculate the weighted entropy
        weightedEntropy = np.sum([(counts[i]/np.sum(counts))*self.entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
        #Calculate the information gain
        informationGain = totalEntropy - weightedEntropy
        # Save calculated information gain to cache if enabled
        if ENABLE_IG_CACHE:
            self.solutionsIgCache[split_attribute_name] = informationGain
        return informationGain



class TabuSearch:
    """
    Tabu Search implementation
    """
    def __init__(self, dataset):
        """
        Initialization method
        """
        self.tabuMemory = deque(maxlen = TABU_LIST_SIZE)
        self.id3 = ID3()
        self.aspirationMemory = set()
        self.data = dataset
        print("Tabu Search initialized...")
        
    def isTerminationCriteriaMet(self, solution):
        """
        Termination criteria is validating the score against a configured value, which can be set to 0 to disable
        """
        if ACCEPT_THRESHOLD > 0 and self.solutionFitness(solution) > ACCEPT_THRESHOLD:
            if ENABLE_EXEC_LOG:
                print("Termination criteria is met")
            return True
        return False

    def solutionFitness(self, solution):
        """
        Solution fitness score is calculated as a total of all information gains for each selected feature
        """
        return sum([self.id3.infoGain(self.data, feature, "B") for feature in solution])

    def aspirationCriteria(self, solution):
        """
        A solution meets the aspiration criteria if it was never tried before
        """
        if ENABLE_EXEC_LOG and not solution in self.aspirationMemory:
            f = random.sample(list(solution), 3)
            # print("Aspiration criteria met with features {}, {}, {}...".format(f[0], f[1], f[2]))
        return not solution in self.aspirationMemory

    def tabuCriteria(self, solution):
        """
        Verifyes if a solution is not tabooed
        """
        if ENABLE_EXEC_LOG and solution in self.tabuMemory:
            f = random.sample(list(solution), 3)
            print("Tabooed solution with features {}, {}, {}...".format(f[0], f[1], f[2]))
        return not solution in self.tabuMemory

    def memorize(self, solution):
        """
        Memorizes current solution for further verification of tabu and aspiration criterias
        """
        self.tabuMemory.append(solution)
        self.aspirationMemory.add(frozenset(solution))

    def putativeNeighbors(self, solution, features):
        """
        Finds TS_MOVES putative solutions that satisfy aspiration and tabu criteria
        """
        putativeSolutions = [None] * TS_MOVES
        idx = 0
        count = 0
        while idx < TS_MOVES:
            count += 1
            if count > LOOP_TRIES:
                print("Unable to find more than {} neighbors".format(idx + 1))
            # Create a random altered solution off by only one feature from the current one
            alteredSolution = self.alterSolution(solution, features)
            # Save newly created solution if it satisfies aspiration criteria
            if self.aspirationCriteria(alteredSolution):
                putativeSolutions[idx] = alteredSolution
                idx += 1
            # Save newly created solution if it isn't tabooed or a duplicate of already saved new solution
            elif not putativeSolutions.__contains__(alteredSolution) and self.tabuCriteria(alteredSolution):
                putativeSolutions[idx] = alteredSolution
                idx += 1
        return putativeSolutions
    
    def alterSolution(self, solution, features):
        """
        Generates a new solution based on the provided by changing only one feature
        """
        alteredSolution = solution.copy()
        # Prepare the pool of features that can be used in new solution
        other_features = set(features) - solution
        # Remove a random feature
        alteredSolution.remove(random.choice(list(solution)))
        # Add a random feature from the pool
        alteredSolution.add(random.choice(list(other_features)))
        return alteredSolution


    def run(self):
        """
        Performs a run of Tabu Search based on initialized data set and returns the best found solution
        """
        print("Feature selection initiated.")
        if ENABLE_EXEC_LOG:
            print("\nexecution log:")
        features = DataLoader.getFeatures(self.data)
        self.bestSolution = set(random.sample(list(features), SUBSET_SIZE))
        self.memorize(self.bestSolution)
        self.currSolution = self.bestSolution
        
        step = 0
        while not self.isTerminationCriteriaMet(self.currSolution) and step < MAX_MOVES:
            step += 1
            # Get putative neighbors
            neighbors = self.putativeNeighbors(self.currSolution, features)
            bestFit = 0
            # Find the best solution from putative neighbors and makes it the current one
            for solution in neighbors:
                if self.solutionFitness(solution) > bestFit:
                    self.currSolution = solution
                    bestFit = self.solutionFitness(solution)
            # Memorize the current solution
            self.memorize(self.currSolution)
            # Verify if current solution is better than the best one, and saves the current as best, if true
            if self.solutionFitness(self.currSolution) > self.solutionFitness(self.bestSolution):
                self.bestSolution = self.currSolution

        # Return the best solution found
        return self.bestSolution


# Main execution for demonstration on a given dataset
def main(filePath):
    print("***************************************************")
    print("*                TABU SEARCH DEMO                 *")
    print("***************************************************")
    # Load data set
    dataset = DataLoader.load('dataset.txt')
    # Initialize Tabu Search instance
    ts = TabuSearch(dataset)
    # Run the feature selection algorithm
    topSolution = DataLoader.sortFeatureSet(ts.run())
    print("\nBest solution with score {} :\n{}\n".format(ts.solutionFitness(topSolution), topSolution))


if __name__ == "__main__":
    main(sys.argv[1])