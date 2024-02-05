from typing import List, Type, Tuple
from enum import Enum
import numpy as np


# the value is used to correspond to the index in the observation_matrix,
# so must start with 0
class Evidence(Enum):
    true = 0
    false = 1


# the value is used to correspond to the index in the transitional_matrix,
# so must start with 0
class State(Enum):
    healthy = 0
    fever = 1


class HMM:
    def __init__(
        self,
        states_space: Type[State],
        evidence_space: Type[Evidence],
        transitional_matrix: np.ndarray,
        sensor_model: np.ndarray,
        prior: np.ndarray,
    ) -> None:
        self.states_space = states_space
        self.evidence_space = evidence_space
        self.transitional_matrix = transitional_matrix
        self.observation_matrix = sensor_model
        self.prior = prior

    def forward_step(self, fv: np.ndarray, evidence: Evidence) -> np.ndarray:
        prediction = np.matmul(self.transitional_matrix, fv)  # P(X_t | e_1:t-1)

        sensor_dist = self.observation_matrix[evidence.value]

        new_fv = prediction * sensor_dist

        return new_fv / np.sum(new_fv)

    def forward(self, evs: List[Evidence]) -> np.ndarray:
        fv = self.prior
        for e in evs:
            fv = self.forward_step(fv, e)
        return fv

    def backward_step(self, bv: np.ndarray, ev: Evidence) -> np.ndarray:
        return np.matmul(
            self.transitional_matrix.T, bv * self.observation_matrix[ev.value]
        )

    def forward_backward(self, evs: List[Evidence]) -> List[np.ndarray]:
        N = len(evs)

        fvs: List[np.ndarray] = [self.prior]

        svs: List[np.ndarray] = []

        b: np.ndarray = np.ones(np.shape(self.prior))

        for i in range(N):
            fvs.append(self.forward_step(fvs[i], evs[i]))

        for i in reversed(range(N)):
            svs.append(fvs[i + 1] * b)
            b = self.backward_step(b, evs[i])

        return svs

    def viterbi(self, evs: List[Evidence]) -> Tuple[List[State], np.ndarray]:
        N = len(evs)

        states_graph: List[State] = []

        # initialization
        prob_map = self.transitional_matrix * self.prior

        state_map: List[np.ndarray] = [np.argmax(prob_map, axis=1)]
        most_likely_prob: List[np.ndarray] = [
            self.observation_matrix[evs[0].value] * np.max(prob_map)
        ]

        # recursion
        for i in range(1, N):
            prob_map = self.transitional_matrix * most_likely_prob[i - 1]

            most_likely_prob.append(
                self.observation_matrix[evs[i - 1].value] * np.max(prob_map)
            )
            state_map.append(np.argmax(prob_map, axis=1))

        states_graph.append(self.states_space(int(np.argmax(most_likely_prob[-1]))))

        max_prob = np.max(most_likely_prob[-1])

        for step in reversed(state_map):
            state_idx = step[states_graph[-1].value]
            states_graph.append(self.states_space(state_idx))

        # states need to be reversed
        return states_graph[::-1], max_prob


def test_viterbi():
    evidences = [Evidence.true, Evidence.false, Evidence.true]

    hmm = HMM(
        states_space=State,
        evidence_space=Evidence,
        transitional_matrix=np.array([[0.7, 0.3], [0.3, 0.7]]),
        sensor_model=np.array([[0.9, 0.2], [0.1, 0.8]]),
        prior=np.array([0.5, 0.5]),
    )

    states, prob = hmm.viterbi(evidences)

    print("len of states: ", len(states))
    print("States are: ", states)
    print("prob of this: ", prob)


def test_forward():
    hmm = HMM(
        states_space=State,
        evidence_space=Evidence,
        transitional_matrix=np.array([[0.7, 0.3], [0.3, 0.7]]),
        sensor_model=np.array([[0.9, 0.2], [0.1, 0.8]]),
        prior=np.array([0.5, 0.5]),
    )

    fv = hmm.forward_step(hmm.prior, Evidence.true)

    expected_value = (
        np.matmul(hmm.transitional_matrix, hmm.prior) * hmm.observation_matrix
    ) / np.sum(
        np.matmul(hmm.transitional_matrix, hmm.prior) * hmm.observation_matrix[0]
    )

    # assert np.allclose(
    #     fv,
    #     expected_value,
    #     atol=1e-1,
    # ), f"Expected: {expected_value} but got {fv}"

    print(f"Expected: {expected_value}, Computed: {fv}")

    print("You need manual pass check!")


if __name__ == "__main__":
    # test_forward()
    test_viterbi()
