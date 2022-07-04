import random
import copy
import Levenshtein as lv
from rdkit import Chem
import numpy as np




class Levenshtein_augment():
    def __init__(self, source_augmentation=10, randomization_tries=1000):
        self.source_augmentation = source_augmentation
        self.randomization_tries = randomization_tries


    def randomize_smiles(_, SMILES, tries):
        """
        Randomizes SMILES [tries] times
        The return list contains unique randomized smiles and can thus be smaller than the number of tries
        :param SMILES: SMILES string 
        :type SMILES: string
        :param tries: number of times to permute
        :type n_perms: int
        :return: list of randomized SMILES
        :rtype: list
        """

        # Generate molelcule from SMILES string
        mol = Chem.MolFromSmiles(SMILES)
        # Extract atom numbers
        ans = list(range(mol.GetNumAtoms()))
        randomizations = []

        while len(randomizations) < tries:
            #print(randomization)
            # Permute atom ordering
            np.random.shuffle(ans)
            # Renumber atoms
            rmol = Chem.RenumberAtoms(mol, ans)
            # Convert random mol back to SMILES
            rSMILES = Chem.MolToSmiles(rmol, canonical=False)
            randomizations.append(rSMILES)
        return list(set(randomizations))

    
    def levenshtein_pairing(self, in_smiles, out_smiles):
        """performs randomizations of lists and finds the target sequence most similar to the input sequences"""
        #TODO make larger number of tries for in randomizations + sampling of set to make chance of getting different SMILES higher for small molecules
        in_randomizations = self.randomize_smiles(in_smiles, self.randomization_tries)[0:self.source_augmentation] 
        out_randomizations = self.randomize_smiles(out_smiles, self.randomization_tries)

        pairs = []
        for in_smile in in_randomizations:
            #Find the alignment score bew
            scores = []
            for out_smile in out_randomizations:
                score = 0
                for smile in in_smile.split("."):
                    ratio = lv.ratio(smile, out_smile)
                    score += ratio
                scores.append(score)
            #
            ranks = np.argsort(scores)
            best_idx = ranks[-1]
            best_outsmile = out_randomizations[best_idx]
            pairs.append( ( in_smile, best_outsmile, scores[best_idx]))
        
        return pairs


    def sample_pairs(self, pairs, times=None):
        """Samples pairs list if shorter than target augmentations"""

        if not times:
            times = self.source_augmentation

        #No need to resample list if its the target length
        if len(pairs) == times:
            return pairs

        samples = []
        pairs_copy = []
        #Sample by emptying the list by random selectio without replacement
        #If list gets empty, recreate it.
        while len(samples) < times:
            if not pairs_copy:
                pairs_copy = copy.copy(pairs)
                random.shuffle(pairs_copy)
            samples.append(pairs_copy.pop())

        return samples