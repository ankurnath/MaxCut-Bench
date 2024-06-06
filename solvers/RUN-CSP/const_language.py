import json
import torch


class Constraint_Language:
    """ Class to represent a fixed Constraint Language """

    def __init__(self, domain_size, relations):
        """
        :param domain_size: Size of the underlying domain
        :param relations: A dict specifying the relations of the language. This also specifies a name for each relation.
                          I.E {'XOR': [[0, 1], [1, 0]], 'AND': [[1,1]]}
        """
        self.domain_size = domain_size
        self.relations = {rel: torch.tensor(tup) for rel, tup in relations.items()}

        # compute characteristic matrices for each relation
        self.char_matrices = dict()
        self.is_symmetric = dict()
        for rel, tup in self.relations.items():
            R = torch.zeros((self.domain_size, self.domain_size), dtype=torch.float32)
            R[tup[:, 0], tup[:, 1]] = 1.0
            self.char_matrices[rel] = R
            self.is_symmetric[rel] = torch.allclose(R, R.T, rtol=1e-05, atol=1e-08)

        self.device = 'cpu'

    def to(self, device):
        self.device = device
        for rel in self.relations.keys():
            self.relations[rel] = self.relations[rel].to(device)
            self.char_matrices[rel] = self.char_matrices[rel].to(device)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump({'domain_size': self.domain_size, 'relations': self.relations}, f, indent=4)

    @staticmethod
    def load(path):
        with open(path, 'r') as f:
            data = json.load(f)

        language = Constraint_Language(data['domain_size'], data['relations'])
        return language

    @staticmethod
    def get_coloring_language(num_colors):

        clauses = []
        for i in range(num_colors):
            for j in range(num_colors):
                if not i == j:
                    clauses.append([i, j])

        lang = Constraint_Language(
            domain_size=num_colors,
            relations={'NEQ': clauses}
        )
        return lang

    @staticmethod
    def get_mis_language():
        lang = Constraint_Language(
            domain_size=2,
            relations={'NAND': [[0, 0], [0, 1], [1, 0]]}
        )
        return lang

    @staticmethod
    def get_2sat_language():
        lang = Constraint_Language(
            domain_size=2,
            relations={
                'OR': [[0, 1], [1, 0], [1, 1]],
                'IMPL': [[0, 0], [0, 1], [1, 1]],
                'NAND': [[0, 0], [0, 1], [1, 0]]
            }
        )
        return lang

    @staticmethod
    def get_maxcut_language():
        lang = Constraint_Language(
            domain_size=2,
            relations={
                'EQ': [[1, 1], [0, 0]],
                'NEQ': [[1, 0], [0, 1]]
            }
        )
        return lang
