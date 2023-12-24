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


cdef class VNodeParticles(VNode):
    """AC_POMCP's VNode maintains particle belief"""
    def __init__(self, num_visits, belief=Particles([])):
        self.num_visits = num_visits
        self.belief = belief
        self.children = {}  # a -> QNode
    def __str__(self):
        return "VNode(%.3f, %.3f, %d | %s)" % (self.num_visits, self.value, len(self.belief),
                                               str(self.children.keys()))
    def __repr__(self):
        return self.__str__()

cdef class RootVNodeParticles(RootVNode):
    def __init__(self, num_visits, history, belief=Particles([])):
        # vnodeobj = VNodeParticles(num_visits, value, belief=belief)
        RootVNode.__init__(self, num_visits, history)
        self.belief = belief
    @classmethod
    def from_vnode(cls, vnode, history):
        rootnode = RootVNodeParticles(vnode.num_visits, history, belief=vnode.belief)
        rootnode.children = vnode.children
        return rootnode

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

    @property
    def update_agent_belief(self):
        ### UNSURE ###
        """True if planner's update function also updates agent's
        belief."""
        return True

    # def updateNetworks(self):
    #     ### ADD NEW ###


    ### NEW ###
    cpdef public plan(self, Agent agent):
        cdef Action action
        cdef float time_taken
        cdef int sims_count

        self._agent = agent   # switch focus on planning for the given agent
        if not hasattr(self._agent, "tree"):
            self._agent.add_attr("tree", None)
        action, time_taken, sims_count = self._search()
        self._last_num_sims = sims_count
        self._last_planning_time = time_taken
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

        start_time = time.time()
        while True:
            ## Note: the tree node with () history will have
            ## the init belief given to the agent.
            state = self._agent.sample_belief()
            
            #### TESTING ####
            # print("During one simulation")
            # print(f"Sampled Particle On Which Sim Is Conditioned {state}")
            # # print(f"Current belief {self._agent._cur_belief}")
            # print(f"History {self._agent.history}")
            # print(f"Search Tree: {self._agent.tree}")
            # if not self._agent.tree == None:
            #     print(f"Action values at root node: {self._agent.tree.print_children_value()}")
            #     print(f"Best action at this point: {self._agent.tree.argmax()}")
            # print("-----")
            #### TESTING ####

            self._simulate(state, self._agent.history, self._agent.tree,
                           None, None, 0)
            sims_count +=1
            time_taken = time.time() - start_time

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

        best_action = self._agent.tree.argmax()
        return best_action, time_taken, sims_count


    cpdef _simulate(AC_POMCP self,
                    State state, tuple history, VNode root, QNode parent,
                    Observation observation, int depth):
        ### CHANGE: Add ac-pomcp logic ###
        total_reward = POUCT._simulate(self, state, history, root, parent, observation, depth)
        if depth == 1 and root is not None:
            root.belief.add(state)  # belief update happens as simulation goes.
        return total_reward


    cpdef update(self, Agent agent, Action real_action, Observation real_observation,
                 state_transform_func=None):
        ### CHANGE ###
        """
        Assume that the agent's history has been updated after taking real_action
        and receiving real_observation.

        `state_transform_func`: Used to add artificial transform to states during
            particle reinvigoration. Signature: s -> s_transformed
        """
        if not isinstance(agent.belief, Particles):
            raise TypeError("agent's belief is not represented in particles.\n"\
                            "AC_POMCP not usable. Please convert it to particles.")
        if not hasattr(agent, "tree"):
            print("Warning: agent does not have tree. Have you planned yet?")
            return

        if agent.tree[real_action][real_observation] is None:
            # Never anticipated the real_observation. No reinvigoration can happen.
            raise ValueError("Particle deprivation.")
        # Update the tree; Reinvigorate the tree's belief and use it
        # as the updated belief for the agent.
        agent.tree = RootVNodeParticles.from_vnode(agent.tree[real_action][real_observation],
                                                   agent.history)
        tree_belief = agent.tree.belief
        agent.set_belief(particle_reinvigoration(tree_belief,
                                                 len(agent.init_belief.particles),
                                                 state_transform_func=state_transform_func))
        # If observation was never encountered in simulation, then tree will be None;
        # particle reinvigoration will occur.
        if agent.tree is not None:
            agent.tree.belief = copy.deepcopy(agent.belief)



    def _VNode(self, agent=None, root=False, **kwargs):
        """Returns a VNode with default values; The function naming makes it clear
        that this function is about creating a VNode object."""
        ### UNSURE ###
        if root:
            # agent cannot be None.
            return RootVNodeParticles(self._num_visits_init,
                                      agent.history,
                                      belief=copy.deepcopy(agent.belief))
        else:
            if agent is None:
                return VNodeParticles(self._num_visits_init,
                                      belief=Particles([]))
            else:
                return VNodeParticles(self._num_visits_init,
                                      belief=copy.deepcopy(agent.belief))
