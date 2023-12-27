"""
This is an implementation of the AC-POMCP algorithm. The algorithm builds on the PO-UCT algorithm which is its base class

This implementation is based on the implementations of other POMDP algorithms from https://h2r.github.io/pomdp-py/html/installation.html
"""

from pomdp_py.framework.basics cimport Action, Agent, POMDP, State, Observation,\
    ObservationModel, TransitionModel, GenerativeDistribution, PolicyModel
from pomdp_py.framework.planner cimport Planner
from pomdp_py.representations.distribution.particles cimport Particles
from pomdp_py.representations.belief.particles cimport particle_reinvigoration
from pomdp_py.algorithms.po_uct cimport VNode, RootVNode, QNode, POUCT, RandomRollout
import copy
import time
import random
import math
from tqdm import tqdm

from .model import dueling_net, MultiHeadAutoencoder


cdef class AC_POMCP(POUCT):

    """AC_POMCP description ..."""

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(), action_prior=None,
                 show_progress=False, pbar_update_interval=5,
                 belief_net=None, q_net=None, env_data_processing=None, init_bel_prob=None):
        super().__init__(max_depth=max_depth,
                         planning_time=planning_time,
                         num_sims=num_sims,
                         discount_factor=discount_factor,
                         exploration_const=exploration_const,
                         num_visits_init=num_visits_init,
                         value_init=value_init,
                         rollout_policy=rollout_policy,
                         action_prior=action_prior,
                         show_progress=show_progress,
                         pbar_update_interval=pbar_update_interval)

        # TODO: torch networks and other classes as cython class variables
        # Defining the belief and q networks and any data processing utils
        # self.belief_net = belief_net
        # self.q_net = q_net
        # self.env_data_processing = env_data_processing
        # self.bel_prob = init_bel_prob


    def updateNetworks(self, hist_conditioned_qvalues):
        # TODO: write network update procedure
        pass

    def getHistoryConditionedQValues(self, bel_state_conditioned_qvalues, probabilities):
        # Get Q(.|h) = SUM (p(s) * Q(.|s))
        # bel_state_conditioned_qvalues is a dictionary of {action: value} pairs for each particle in the belief
        bel_state_conditioned_qvalues = self.env_data_processing.qval_array_from_dict(bel_state_conditioned_qvalues)

        hist_conditioned_qvalues = np.average(bel_state_conditioned_qvalues, axis=0, weights=prob)

        retrun hist_conditioned_qvalues



    ### NEW ###
    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count, hist_conditioned_qvalues = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken


        ### Add update net ###
        return action


    ### NEW ###
    cpdef _search(self):
        cdef State state
        cdef Action best_action
        cdef int sims_count = 0
        cdef float time_taken = 0
        cdef float best_value
        cdef bint stop_by_sims = self._num_sims > 0
        cdef object pbar

        if self._show_progress:
            if stop_by_sims:
                total = int(self._num_sims)
            else:
                total = self._planning_time
            pbar = tqdm(total=total)

        # TODO: torch networks and other classes as cython class variables
        # Compute new belief given the history and old belief
        # belief_tensor = self.env_data_processing.batch_from_particles(belief=self._agent.belief, probabilities=self.bel_prob)
        # cond_tensor = self.env_data_processing.cond_from_history(self._agent.history)
        # self._agent.belief, self.bel_prob = self.belief_net(belief_tensor, cond_tensor)

        bel_state_conditioned_qvalues = []
        for state in self._agent.belief.particles:
            # Run a set of simulations per belief state (stopping criteria could be num_sims or time_taken)

            start_time = time.time()
            while True:
                ## Note: the tree node with () history will have
                ## the init belief given to the agent.

                self._simulate(state, self._agent.history, self._agent.tree,
                               None, None, 0)
                sims_count +=1
                time_taken = time.time() - start_time

                # Refresh the tqdm progress bar
                if self._show_progress and sims_count % self._pbar_update_interval == 0:
                    if stop_by_sims:
                        pbar.n = sims_count
                    else:
                        pbar.n = time_taken
                    pbar.refresh()

                if stop_by_sims:
                    if sims_count >= self._num_sims:
                        break
                else:
                    if time_taken > self._planning_time:
                        if self._show_progress:
                            pbar.n = self._planning_time
                            pbar.refresh()
                        break

                if self._show_progress:
                    pbar.close()

            
            action_values = self._agent.tree.return_children_values()
            bel_state_conditioned_qvalues.append(action_values)


            # best_action = self._agent.tree.argmax()
            self._agent.tree = None

        hist_conditioned_qvalues = self.getHistoryConditionedQValues(bel_state_conditioned_qvalues, self.bel_prob)
        # TODO make a new action or somehow get it from the tree
        best_action = Action()
        
        
        return best_action, time_taken, sims_count, hist_conditioned_qvalues



