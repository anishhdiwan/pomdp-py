import torch
import numpy as np
import pomdp_py
from pomdp_problems.tag.domain.action import MOTION_ACTIONS, TagAction
from pomdp_problems.tag.domain.state import TagState
from pomdp_problems.rocksample import rocksample_problem as rs
import random


### Utilities and convenience methods for neural network related environment data transformations ###

def elegant_pair(input_coords):
    """The Szudzik Function http://szudzik.com/ElegantPairing.pdf
    
    Args:
        input_coords (tuple): x,y position as a tuple of ints
    """
    x = input_coords[0]
    y = input_coords[1]
    print(f"coords xy {x,y}")

    assert isinstance(x, int), "The input must be a tuple of ints"
    assert isinstance(y, int), "The input must be a tuple of ints"

    if x >= y:
        return x*x + x + y
    else:
        return y*y + x


class DataProcessing():
    def __init__(self, t, bel_size):
        self.t = t
        self.bel_size = bel_size

    def cond_from_history(self, history):
    	pass

    def batch_from_particles(self, belief):
    	pass

    def particles_from_output(self, out):
    	pass

    def qval_array_from_dict(self, bel_state_conditioned_qvalues):
        """Returns a numpy array of action values given a list of dictionaries of action values for each belief state

        Args:
            bel_state_conditioned_qvalues (list(dict)): A list of dictionaries of action values, one dict per belief state

        Returns:
            A numpy array of the same action values

        """
        # Reverse the actions dict
        rev_actions = dict((v, k) for k, v in self.actions.items())
        qval_array = np.zeros((len(bel_state_conditioned_qvalues), self.num_actions))
        
        for idx, qval_dict in enumerate(bel_state_conditioned_qvalues):
            actions_encountered = set(qval_dict.keys())
            qvals = np.zeros((self.num_actions))
            for i in range(len(qvals)):
                if rev_actions[i] in actions_encountered:
                    qvals[i] = qval_dict[rev_actions[i]]
                else:
                    qvals[i] = None

            qval_array[idx] = qvals

        return torch.from_numpy(qval_array).to(torch.float)


class TagDataProcessing(DataProcessing):
    """
    Prepare data from the tag env for forwarding to the neural network. Also process network outputs

    
    Args:
        grid_map (tag.models.components.grid_map.GridMap): Holds information of the map occupancy
        t (int): The number of past a,o steps in the truncated history
        bel_size (int): size of the belief subset
        belief (pomdp_py.representations.distribution.Particles): object representing a set of particles. Here, belief.particles 
        corresponds to the _values variable which internally is a list of Tag State objects which internally have the features of 
        the samples fed into the neural network
        probabilities (array): probabilities of each belief state
        history (tuple): agent's history 
    """
    def __init__(self, grid_map, t, bel_size):
        super().__init__(t, bel_size)
        self.grid_map = grid_map
        self.num_actions = 5

        # Action name must be the same as the `name` field of the action class
        self.actions = {
        "move-xy-North2D": 0,
        "move-xy-South2D": 1,
        "move-xy-East2D": 2,
        "move-xy-West2D": 3,
        "tag": 4, 
        }

        actions_set = MOTION_ACTIONS | set({TagAction()}) # | is a set union operator  
        self.actions_dict = dict()
        for item in actions_set:
            self.actions_dict[item.name] = item


    def cond_from_history(self, history):
        """Generate a conditioning vector given the history

        Args:
            history (tuple): A tuple of repeated action, observation data seen in the real environment 
            action (tag domains.Action):  Action taken, encoded as an int in [0, num_actions] in this function
            observation (tag domain .Observation): Observation seen in the env, encoded as a float in [-1, elegent_pair(max_x, max_y)] in this function

        Returns:
            Torch tensor of some t step history. Necessary padding applied if len(history) < t

        """
        # history = ((a,o), ...)
        # Flatten to one list
        history_flat = []
        for hist_tuple in history:
            for entry in hist_tuple:
                history_flat.append(entry)
        history = history_flat


        # Convert history objects to floats
        for idx, entry in enumerate(history):
            if isinstance(entry, pomdp_py.Action):
                history[idx] = self.actions[entry.name]

            if isinstance(entry, pomdp_py.Observation):
                # Observation is a tuple of Target pose (x,y). 
                # Using the Elegant Pair Fn. to map to a 1D space
                if entry.target_position == None:
                    history[idx] = -1.
                else:
                    history[idx] = elegant_pair(entry.target_position)

        
        # Padding/slicing
        if len(history) == self.t:
            return torch.from_numpy(np.array(history)).to(torch.float)

        if len(history) < self.t:
            # Zero padding if the history tuple is smaller than t
            hist_tensor = torch.zeros((self.t))
            hist_tensor[(self.t - len(history)):] = torch.from_numpy(np.array(history))
            return hist_tensor.to(torch.float)

        if len(history) > self.t:
            return torch.from_numpy(np.array(history[-self.t:])).to(torch.float)


    def batch_from_particles(self, belief):
        """Generate a batch of tensors with input samples for the belief network

        Args:
            belief (pomdp_py.representations.distribution.Particles): A set of particles. Here, belief.particles 
            corresponds to the _values variable which internally is a list of tag.State objects which internally have the features of 
            the samples

        Returns:
            Torch tensor for a batch representing the belief state features

        """
        assert len(belief.particles) == self.bel_size, "The number of particles must match the size of the belief subset"  
        batch = torch.zeros(self.bel_size, 5) # Currently this is configured only for the default tag world. Changes to the world need changes here

        for idx, particle in enumerate(belief.particles):
            sample_robot_position = particle.robot_position # tuple
            sample_target_position = particle.target_position # tuple

            sample_target_found = particle.target_found # bool
            sample_target_found = 1. if sample_target_found == True else 0.

            sample = np.concatenate((np.array(sample_robot_position), np.array(sample_target_position), sample_target_found), axis=None)
            batch[idx] = torch.from_numpy(sample).to(torch.float)

        return batch


    def particles_from_output(self, out):
        """Generate a belief instance (pomdp_py.representations.distribution.Particles) from the output of the neural network

        Reinvigorate particles if the network predicted a belief subset with the same particles repeated

        Args:
            out (tensor): A tensor of the output of the neural network


        Returns:
            A belief (pomdp_py.representations.distribution.Particles) instance representing the outputs of the neural network

        """
        out = out.detach().numpy()
        particles = []
        for i in range(self.bel_size):
            sample = out[i]
            sample_robot_position = (int(round(sample[0])), int(round(sample[1])))
            sample_target_position = (int(round(sample[2])), int(round(sample[3])))

            sample_target_found = sample[-1]
            sample_target_found = True if sample_target_found < 0.5 else False           

            particles.append(TagState(sample_robot_position, sample_target_position, sample_target_found))


        # Drop duplicates
        particles = list(set(particles))
        # print(f"\n num unique predicted particles {len(particles)}")

        # Drop particles that are infeasible (in wrong parts of the state space)
        for particle in particles:
            if (particle.target_position in self.grid_map.obstacle_poses) or (particle.robot_position in self.grid_map.obstacle_poses):
                particles.pop(particle)
        # print(f"\n num predicted particles after removing invalids {len(particles)}")

        # It is possible that the neural network returns a new belief such that a few particles are repeated. This is okay. However, for 
        # better exploration, new particles are added whenever this happens
        num_predicted_particles = len(particles) 
        num_particles_to_add = self.bel_size - num_predicted_particles 
        while num_particles_to_add > 0:
            sample_robot_position = (random.randint(0, self.grid_map.width-1),
                               random.randint(0, self.grid_map.length-1))
            sample_target_position = (random.randint(0, self.grid_map.width-1),
                               random.randint(0, self.grid_map.length-1))
            if (sample_robot_position in self.grid_map.obstacle_poses) or (sample_target_position in self.grid_map.obstacle_poses):
                # Skip obstacles
                continue            

            particles.append(TagState(sample_robot_position, sample_target_position, False))
            particles = list(set(particles)) # Redefine particles just in case the added one was a duplicate
            num_particles_to_add = self.bel_size - len(particles) # Update counter

        belief = pomdp_py.Particles(particles)
        return belief


class RocksampleDataProcessing(DataProcessing):
    """
    Prepare data from the rocksample env for forwarding to the neural network. Also process network outputs

    
    Args:
        n (int): size of the rocksample env
        k (int): number of rocks in the env
        t (int): The number of past a,o steps in the truncated history
        bel_size (int): size of the belief subset
        belief (pomdp_py.representations.distribution.Particles): object representing a set of particles. Here, belief.particles 
        corresponds to the _values variable which internally is a list of RockSample State objects which internally have the features of 
        the samples fed into the neural network
        probabilities (array): probabilities of each belief state
        history (tuple): agent's history 
    """
    def __init__(self, n, k, t, bel_size):
        super().__init__(t, bel_size)
        self.n = n
        self.k = k
        self.num_fixed_actions = 5
        self.num_actions = k + 5

        self.actions = {
        "move-NORTH": 0,
        "move-SOUTH": 1,
        "move-EAST": 2,
        "move-WEST": 3,
        "sample": 4, 
        }
        for i in range(self.k):
            self.actions[f"check-{i}"] = int(i + self.num_fixed_actions)

        check_actions = set({rs.CheckAction(rock_id) for rock_id in range(self.k)})
        _move_actions = {rs.MoveEast, rs.MoveWest, rs.MoveNorth, rs.MoveSouth}
        _other_actions = {rs.SampleAction()} | check_actions
        _all_actions = _move_actions | _other_actions

        self.actions_dict = dict()
        for item in _all_actions:
            self.actions_dict[item.name] = item


    def cond_from_history(self, history):
        """Generate a conditioning vector given the history

        Args:
            history (tuple): A tuple of repeated action, observation data seen in the real environment 
            action (rocksample_problem.Action):  Action taken, encoded as an int in range[0, 6] in this function
            observation (rocksample_problem.Observation): Observation seen in the env, encoded as a float in [0., 1.] in this function

        Returns:
            Torch tensor of some t step history. Necessary padding applied if len(history) < 10

        """
        # history = ((a,o), ...)
        # Flatten to one list
        history_flat = []
        for hist_tuple in history:
            for entry in hist_tuple:
                history_flat.append(entry)
        history = history_flat

        # assert len(history) < self.t, "t (the size of the tensor of history) must not be less than the size of the history"

        # Convert history objects to floats
        for idx, entry in enumerate(history):
            if isinstance(entry, pomdp_py.Action):
                history[idx] = self.actions[entry.name]

            if isinstance(entry, pomdp_py.Observation):
                if entry.quality == None:
                    history[idx] = 0.5
                elif entry.quality == 'good':
                    history[idx] = 1.0
                elif entry.quality == 'bad':
                    history[idx] = 0.0

        if len(history) == self.t:
            return torch.from_numpy(np.array(history)).to(torch.float)

        if len(history) < self.t:
            # Zero padding if the history tuple is smaller than t
            hist_tensor = torch.zeros((self.t))
            hist_tensor[(self.t - len(history)):] = torch.from_numpy(np.array(history))
            return hist_tensor.to(torch.float)

        if len(history) > self.t:
            return torch.from_numpy(np.array(history[-self.t:])).to(torch.float)


    def batch_from_particles(self, belief):
        """Generate a batch of tensors with input samples for the belief network

        Args:
            belief (pomdp_py.representations.distribution.Particles): A set of particles. Here, belief.particles 
            corresponds to the _values variable which internally is a list of rocksample_problem.State objects which internally have the features of 
            the samples
            probabilities (array): Probabilities of each belief state

        Returns:
            Torch tensor for a batch representing the belief state features

        """
        assert len(belief.particles) == self.bel_size, "The number of particles must match the size of the belief subset"  
        batch = torch.zeros(self.bel_size,self.k+3)
        # if torch.is_tensor(probabilities):
        #     probabilities = probabilities.detach().numpy()

        for idx, particle in enumerate(belief.particles):
            sample_pos = particle.position
            sample_rocktypes = list(particle.rocktypes)
            for i in range(len(sample_rocktypes)):
                sample_rocktypes[i] = 0. if sample_rocktypes[i] == "bad" else 1.
            sample_terminal = particle.terminal
            sample_terminal = 1. if sample_terminal == True else 0.
            # sample_prob = probabilities[idx]


            # sample = np.concatenate((np.array(sample_pos), np.array(sample_rocktypes), sample_terminal, sample_prob), axis=None)
            sample = np.concatenate((np.array(sample_pos), np.array(sample_rocktypes), sample_terminal), axis=None)
            batch[idx] = torch.from_numpy(sample).to(torch.float)

        return batch


    def particles_from_output(self, out):
        """Generate a belief instance (pomdp_py.representations.distribution.Particles) from the output of the neural network

        Reinvigorate particles if the network predicted a belief subset with the same particles repeated

        Args:
            out (tensor): A tensor of the output of the neural network


        Returns:
            A belief (pomdp_py.representations.distribution.Particles) instance representing the outputs of the neural network

        """
        out = out.detach().numpy()
        particles = []
        for i in range(self.bel_size):
            sample = out[i]
            sample_pos = (int(round(sample[0])), int(round(sample[1])))
            sample_rocktypes = sample[2:self.k+2]
            sample_rocktypes = sample_rocktypes > 0.5

            rocktypes = []

            for idx, rocktype in enumerate(sample_rocktypes):
                if rocktype == True:
                    rocktypes.append("good")
                elif rocktype == False:
                    rocktypes.append("bad")

            rocktypes = tuple(rocktypes)
            sample_terminal = sample[-1]
            sample_terminal = True if sample_terminal < 0.5 else False

            particles.append(rs.State(sample_pos, rocktypes, sample_terminal))


        # Drop duplicates
        particles = list(set(particles))

        # It is possible that the neural network returns a new belief such that a few particles are repeated. This is okay. However, for 
        # better exploration, new particles are added whenever this happens
        num_predicted_particles = len(set(particles)) 
        num_particles_to_add = self.bel_size - num_predicted_particles 
        if num_particles_to_add > 0:
            for _ in range(num_particles_to_add):
                rocktypes = []
                for i in range(self.k):
                    rocktypes.append(rs.RockType.random())
                rocktypes = tuple(rocktypes)
                particles.append(rs.State((random.randint(0, self.n-1), random.randint(0, self.n-1)), rocktypes, False))

            # # Keeping the first probability and repeating the others
            # probabilities[num_predicted_particles:] = np.full((num_particles_to_add), (1-np.sum(probabilities[:num_predicted_particles]))/num_particles_to_add)

        belief = pomdp_py.Particles(particles)

        return belief