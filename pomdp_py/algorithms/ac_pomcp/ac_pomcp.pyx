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
        """
        Instead of directly returning the best action, the plan() method from POUCT is modified to return action values conditioned on the belief state.

        History conditioned action values are computed out of the loop with other vectors computed through a parameterised function
        """
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)

        bel_state_conditioned_qvalues, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken

        return bel_state_conditioned_qvalues


    ### NEW ###
    cpdef _search(self):
        cdef State state
        cdef Action best_action
        cdef int sims_count = 0
        # Defining another counter to count the number of simulations done under each belief state
        cdef int bel_conditioned_sims_count = 0
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
            bel_conditioned_sims_count = 0

            start_time = time.time()
            while True:
                # print(f"Search Tree: {self._agent.tree}")
                ## Note: the tree node with () history will have
                ## the init belief given to the agent.

                self._simulate(state, self._agent.history, self._agent.tree,
                               None, None, 0)
                sims_count +=1
                bel_conditioned_sims_count +=1
                time_taken = time.time() - start_time

                # Refresh the tqdm progress bar
                if self._show_progress and sims_count % self._pbar_update_interval == 0:
                    if stop_by_sims:
                        pbar.n = sims_count
                    else:
                        pbar.n = time_taken
                    pbar.refresh()

                if stop_by_sims:
                    if bel_conditioned_sims_count >= self._num_sims:
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


            # Reset the tree for the next belief state
            self._agent.tree = None


        return bel_state_conditioned_qvalues, time_taken, sims_count 



    cpdef public update(self, Agent agent, Action real_action, Observation real_observation):
        """
        Add the real_action and real_observation to agent.tree (which is currently None). Then prune all other branches 
        And set agent.tree[real_action][real_observation] as the new root node

        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.
        """

        # Define the tree as a new root node with the previous (un-updated) history 
        if agent.tree is None:
            root = RootVNode(self._num_visits_init, agent.history[:-1])
            agent.tree = root

        # Expand the new tree with real_action and real_observation
        if agent.tree[real_action] is None:
            selected_action = QNode(self._num_visits_init,
                                        self._value_init)
            agent.tree[real_action] = selected_action

        if agent.tree[real_action][real_observation] is None:
            agent.tree[real_action][real_observation] =  self._VNode()


        if not hasattr(agent, "tree") or agent.tree is None:
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if real_action not in agent.tree\
           or real_observation not in agent.tree[real_action]:
            agent.tree = None  # replan, if real action or observation differs from all branches
        elif agent.tree[real_action][real_observation] is not None:
            # Update the tree (prune)
            agent.tree = RootVNode.from_vnode(
                agent.tree[real_action][real_observation],
                agent.history)
        else:
            raise ValueError("Unexpected state; child should not be None")