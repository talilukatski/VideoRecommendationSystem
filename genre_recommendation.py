import numpy as np


class Recommender:

    def __init__(self, L, S, p):
        """_summary_

        Args:
        L (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                         will give a like to a clip from genre i.
        S (np.ndarray): A matrix in which the entry (i,j) represents the probability that a user of type j
                        won't leave the system after being recommended a clip from genre i and not liking it.
        p (np.ndarray): The prior over user types. The entry i represents the probability that a user is of type i."""
        self.L = L.copy()
        self.S = S.copy()
        self.p = p.copy()

        # If S = 0, we want to remove rows that contain only identical probabilities (e.g., [0.7,0.7,0.7]),
        # but only if there is at least one other row where 2/3 of probabilities are greater than
        # the repeated probability (e.g., [0.8,0.2,0.9]).
        if np.all(self.S == 0):
            delete_rows = []
            row_num, col_num = self.L.shape
            for i in range(row_num):
                if np.all(self.L[i] == self.L[i][0]):  # check if row contains only identical probs
                    prob = L[i][0]
                    for j in range(row_num):
                        if i != j:
                            # check if the row have 2/3 prob greater than the identical prob
                            cnt = np.sum(self.L[j] > prob)
                            if cnt >= col_num * 2 / 3:
                                delete_rows.append(i)
                                break
            self.L = np.delete(self.L, delete_rows, axis=0)

    def recommend(self):
        """_summary_

        Returns:
        integer: The index of the clip that the recommender recommends to the user."""
        if np.all(self.S == 0):
            recommends = np.dot(self.L, self.p)
        else:
            recommends = 0.7 * np.dot(self.L, self.p) + 0.3 * np.dot(self.S, self.p)
        self.genre = np.argmax(recommends)
        return self.genre

    def update(self, signal):
        """_summary_

        Args:
        signal (integer): A binary variable that represents whether the user liked the recommended clip or not.
                          It is 1 if the user liked the clip, and 0 otherwise."""
        if signal == 1:
            self.p *= self.L[self.genre]
        else:
            self.p *= (1 - self.L[self.genre]) * self.S[self.genre]
        self.p /= np.sum(self.p)
