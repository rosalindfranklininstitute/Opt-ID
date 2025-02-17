# Copyright 2017 Diamond Light Source
# 
# Licensed under the Apache License, Version 2.0 (the "License"); 
# you may not use this file except in compliance with the License. 
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, 
# software distributed under the License is distributed on an 
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
# either express or implied. See the License for the specific 
# language governing permissions and limitations under the License.

'''
Created on 9 Dec 2013

@author: ssg37927
'''

import os
import copy
import pickle
import binascii

import copy
import random
import numpy as np

from .field_generator import calculate_trajectory_loss_from_array, \
                             calculate_cached_trajectory_loss,     \
                             generate_per_magnet_array,            \
                             compare_magnet_arrays

from .logging_utils import logging, getLogger
logger = getLogger(__name__)


class BCell(object):

    def __init__(self, available=None):
        self.age = 0
        self.fitness = None
        self.genome = None
        self.uid = binascii.hexlify(os.urandom(6)).decode()
        self.available = available

    def clone(self):
        return copy.deepcopy(self)

    def create(self, *args, **kargs):
        raise Exception("create needs to be implemented")

    def age_bcell(self):
        self.age += 1

    def save(self, path):
        filename = '%010.8e_%03i_%s.genome' %(self.fitness, self.age, self.uid)

        with open(os.path.join(path, filename), 'wb') as fp:
            pickle.dump(self.genome, fp)

    def load(self, filename):

        with open(filename, 'rb') as fp:
            self.genome = pickle.load(fp)

        params = os.path.split(filename)[1].split('_')
        self.fitness = float(params[0])
        self.age = int(params[1])
        self.uid = params[2].split('.')[0]

    def generate_children(self, *args, **kargs):
        raise Exception("generate_children needs to be implemented")


class ID_BCell(BCell):

    def __init__(self, available=None):
        super().__init__(available=available)
        self.mutations = 0

    def create(self, info, lookup, magnets, maglist, ref_trajectories):
        self.genome = maglist
        _, self.fitness = calculate_cached_trajectory_loss(info, lookup, magnets, self.genome, ref_trajectories)

    def generate_children(self, number_of_children, number_of_mutations, info, lookup, magnets, ref_trajectories):
        # Increment the age of the parent genome
        self.age_bcell()

        # Evaluate the parent genome and calculate its bfield
        # TODO this can be cached on genome creation incase this genome lives through multiple generations
        parent_bfield, trajectory_loss = calculate_cached_trajectory_loss(info, lookup, magnets, self.genome, ref_trajectories)
        parent_per_magnet_array        = generate_per_magnet_array(info, self.genome, magnets)

        logger.debug('Estimated fitness to real fitness error %1.8E', abs(self.fitness - trajectory_loss))
        self.fitness = trajectory_loss

        # Sample a set of child genomes mutated from the current parent
        children = []
        for genome_index in range(number_of_children):

            # Clone the current genome and apply a number of random mutations
            child_magnet_lists = copy.deepcopy(self.genome)
            child_magnet_lists.mutate(number_of_mutations, available=self.available)

            # Calculate the bfield of the child genome w.r.t to the parent one for efficiency
            child_per_magnet_array  = generate_per_magnet_array(info, child_magnet_lists, magnets)
            per_beam_bfield_updates = compare_magnet_arrays(parent_per_magnet_array, child_per_magnet_array, lookup)
            child_bfield  = parent_bfield - sum(per_beam_bfield_updates.values())
            child_fitness = calculate_trajectory_loss_from_array(info, child_bfield, ref_trajectories)

            # Create the child genome object
            genome = ID_BCell(available=self.available)
            genome.mutations = number_of_mutations
            genome.genome    = child_magnet_lists
            genome.fitness   = child_fitness
            children.append(genome)

            logger.debug('Created child genome %d of %d with fitness %1.8E',
                         genome_index, number_of_children, genome.fitness)

        return children

# TODO ID_Shim_BCell is marked for deprecation and removal along with the mpi_runner_for_shim_opt.py script
#      due to data dependency on the initial parent genome (due to reference holding) and required determinism
#      of the RNG and function call order
class ID_Shim_BCell(BCell):

    def __init__(self, available=None):
        BCell.__init__(self, available=available)
        self.mutations = 0

    def create(self, info, lookup, magnets, maglist, ref_trajectories, number_of_changes, original_bfield):
        self.magnets = magnets #added 22/03/19 ZP
        self.maglist = maglist
        self.changes = number_of_changes
        self.create_genome(number_of_changes, available=self.available)
        
        maglist = copy.deepcopy(self.maglist)
        maglist.mutate_from_list(self.genome)
        new_magnets = generate_per_magnet_array(info, maglist, magnets)
        original_magnets = generate_per_magnet_array(info, self.maglist, magnets)
        update = compare_magnet_arrays(original_magnets, new_magnets, lookup)
        updated_bfield = np.array(original_bfield)
        
        for beam in update.keys() :
            if update[beam].size != 0:
                updated_bfield = updated_bfield - update[beam]
        self.fitness = calculate_trajectory_loss_from_array(info, updated_bfield, ref_trajectories)

#     Hardcoded numbers! Based on length of sim file available for shimming! Warning!
#     Removed hardcoded numbers, available based on magnet input file. 18/02/19 ZP+MB
    def create_genome(self, number_of_mutations, available=None):
#   def create_genome(self, number_of_mutations, available={'HH':range(102)}):
#   def create_genome(self, number_of_mutations, available={'VE':range(20), 'HE':range(20), 'HH':range(420), 'VV':range(419)}):
#   def create_genome(self, number_of_mutations, available={'HH':(range(36,44,1)+range(22,27,1)+range(226,234,1)+range(212,217,1)+range(382,420,1)), 'VV':(range(36,44,1)+range(22,27,1)+range(226,234,1)+range(212,217,1)+range(382,419,1))}):
        
        if (available == None):
            #available = self.maglist.availability() maglist doesn't have an availability attribute
            available = self.available if (self.available is not None) else self.magnets.availability() #added 22/03/19 ZP
            logging.debug("availability is %s"%(available))

        self.genome = []
        for i in range(number_of_mutations):
            # pick a list at random
            key = random.choice(list(available.keys()))
            # pick a flip or swap
            p1 = random.choice(available[key])
            p2 = random.choice(available[key])
            
            if random.random() > 0.5 :
                # swap
                self.genome.append(('S', key, p1, p2))
            else :
                self.genome.append(('F', key, p1, p2))

#     Hardcoded numbers! Based on length of sim file available for shimming! Warning!
#     Removed hardcoded numbers, available based on magnet input file. 18/02/19 ZP+MB
    def create_mutant(self, number_of_mutations, available=None):
#    def create_mutant(self, number_of_mutations, available={'VE':range(12), 'HE':range(12), 'HH':range(420), 'VV':range(419)}):
    #def create_mutant(self, number_of_mutations, available={'HH':(range(36,44,1)+range(22,27,1)+range(226,234,1)+range(212,217,1)+range(382,420,1)), 'VV':(range(36,44,1)+range(22,27,1)+range(226,234,1)+range(212,217,1)+range(382,419,1))}):
        if (available == None):
            #available = self.maglist.availability()
            available = self.available if (self.available is not None) else self.magnets.availability()
        
        mutant = copy.deepcopy(self.genome)
        for i in range(number_of_mutations):
            position = random.randint(0, len(mutant)-1)
            # pick a list at random
            key = random.choice(list(available.keys()))
            # pick a flip or swap
            p1 = random.choice(available[key])
            p2 = random.choice(available[key])
            
            if random.random() > 0.5 :
                # swap
                mutant[position] = ('S', key, p1, p2)
            else :
                mutant[position] = ('F', key, p1, p2)
        return mutant


    def generate_children(self, number_of_children, number_of_mutations, info, lookup, magnets, ref_trajectories, real_bfield=None):
        # first age, as we are now creating children
        self.age_bcell()
        children = []
        
        # Generate the IDfiled for the parent, as we need to calculate it fully here.
        original_bfield = None
        original_fitness = None
        if real_bfield is None:
            original_bfield, original_fitness = calculate_cached_trajectory_loss(info, lookup, magnets, self.genome, ref_trajectories)
            fitness_error = abs(self.fitness - original_fitness)
            logging.debug("Estimated fitness to real fitness error %2.10e"%(fitness_error))

        else :
            original_bfield = real_bfield
            original_fitness = calculate_trajectory_loss_from_array(info, original_bfield, ref_trajectories)
            fitness_error = abs(self.fitness - original_fitness)
            logging.debug("Using real bfield")
        
        original_magnets = generate_per_magnet_array(info, self.maglist, magnets)
        maglist = copy.deepcopy(self.maglist)
        mutation_list = self.create_mutant(number_of_mutations, available=self.available)
        maglist.mutate_from_list(mutation_list)
        new_magnets = generate_per_magnet_array(info, maglist, magnets)
        update = compare_magnet_arrays(original_magnets, new_magnets, lookup)
        updated_bfield = np.array(original_bfield)
        for beam in update.keys() :
            if update[beam].size != 0:
                updated_bfield = updated_bfield - update[beam]
        self.fitness = calculate_trajectory_loss_from_array(info, updated_bfield, ref_trajectories)

        for i in range(number_of_children):
            maglist = copy.deepcopy(self.maglist)
            mutation_list = self.create_mutant(number_of_mutations, available=self.available)
            maglist.mutate_from_list(mutation_list)
            new_magnets = generate_per_magnet_array(info, maglist, magnets)
            update = compare_magnet_arrays(original_magnets, new_magnets, lookup)
            child = ID_Shim_BCell(available=self.available)
            child.mutations = number_of_mutations
            child.genome = mutation_list
            child.maglist = self.maglist
            child.magnets = self.magnets
            updated_bfield = np.array(original_bfield)
            for beam in update.keys() :
                if update[beam].size != 0:
                    updated_bfield = updated_bfield - update[beam]
            child.fitness = calculate_trajectory_loss_from_array(info, updated_bfield, ref_trajectories)
            children.append(child)
            logging.debug("Child created with fitness : %2.10e" % (child.fitness))
        return children
