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


cdef class AC_POMCP(POUCT):

    """AC_POMCP description ..."""

    def __init__(self,
                 max_depth=5, planning_time=-1., num_sims=-1,
                 discount_factor=0.9, exploration_const=math.sqrt(2),
                 num_visits_init=0, value_init=0,
                 rollout_policy=RandomRollout(), action_prior=None,
                 show_progress=False, pbar_update_interval=5):
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


    ### NEW ###
    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, action_value, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken

        return action, action_value


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

        # TODO make a new action or somehow get it from the tree
        # TODO somehow send this outside the class
        hist_conditioned_qvalues = self.getHistoryConditionedQValues(bel_state_conditioned_qvalues, self.bel_prob)
        
        best_action = Action()        
        best_action_value = best_action.value
        return best_action, best_action_value, time_taken, sims_count 



