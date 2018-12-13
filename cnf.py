import random
import itertools
import numpy as np
from numpy import linalg
from scipy.optimize import linprog
import z3

def choose_random_polarity():
  return random.choice([True, False])

def is_ktuple_even(ktuple):
  varlist = [list(d.keys()) for d in ktuple]
  flattened_varlist = [item for sublist in varlist for item in sublist]
  uniq_vars = set(flattened_varlist)
  var_counts = [flattened_varlist.count(s) for s in uniq_vars]
  return all(c % 2 == 0 for c in var_counts)

def is_ktuple_inconsistent(ktuple):
  polarity_list = [list(d.values()) for d in ktuple]
  flattened_polarities = [item for sublist in polarity_list for item in sublist]
  return flattened_polarities.count(False) % 2 == 1

def z3_literal(var,polarity):
  if polarity:
    return z3.Bool('x' + str(var))
  else:
    return z3.Not(z3.Bool('x' + str(var)))

class ThreeCNF:
  def __init__(self,n,m):
    self.n = n
    self.m = m
    self.variables = [i for i in range(n)]
    # a clause is a dictionary with 3 entries mapping variable to polarity
    self.clauses = []

  def get_random_clause(self):
    # pick 3 variables from n without replacement
    clause_vars = random.sample(self.variables,3)
    return {clause_vars[i]: choose_random_polarity() for i in range(3)}

  def get_random_phi(self):
    self.clauses = [self.get_random_clause() for i in range(self.m)]

  def find_all_k_tuples(self,k):
    collection = []
    for indices in itertools.combinations_with_replacement(range(self.m),k):
      tuple = [self.clauses[i] for i in indices]
      if is_ktuple_even(tuple) and is_ktuple_inconsistent(tuple):
        collection.append(indices)
    return collection

  # this is really just for testing
  # and is also wrong i think
  def find_all_four_tuples(self):
    # list of indexes to clauses that form inconsistent even k-tuples
    collection = []
    for i in range(self.m):
      for j in range(self.m):
        for k in range(self.m):
          for l in range(self.m):
            tuple = [self.clauses[i],self.clauses[j],self.clauses[k],self.clauses[l]]
            if is_ktuple_even(tuple) and is_ktuple_inconsistent(tuple):
              collection.append([i,j,k])
    return collection

  def calculate_m_phi(self):
    m_phi = np.zeros((self.n,self.n))
    for i in range(self.m):
      clause_vars = list(self.clauses[i].keys())
      for j in clause_vars:
        for k in clause_vars:
          if j != k:
            if self.clauses[i][j] == self.clauses[i][k]:
              update_val = -0.5
            else:
              update_val = 0.5
            m_phi[j][k] += update_val
    return m_phi

  def find_m_phi_eigenvalue(self):
    m_phi = self.calculate_m_phi()
    eigenvalues = linalg.eigvals(m_phi)
    return max(eigenvalues)

  def find_imbalance(self):
    positive_vars = [0] * self.n
    negative_vars = [0] * self.n
    for clause in self.clauses:
      for k in clause:
        if clause[k]:
          positive_vars[k] += 1
        else:
          negative_vars[k] += 1
    return sum([abs(positive_vars[i] - negative_vars[i]) for i in range(self.n)])

  def build_lp(self,k,d):
    ktuples = self.find_all_k_tuples(k)
    T = len(ktuples)
    c = np.ones(T) * -1
    A_ub = np.zeros((self.m,T))
    for clause_idx in range(self.m):
      for t_idx in range(T):
        if clause_idx in ktuples[t_idx]:
          A_ub[clause_idx][t_idx] = 1
        else:
          A_ub[clause_idx][t_idx] = 0
    b_ub = np.full(self.m,d)
    res = linprog(c,A_ub=A_ub,b_ub=b_ub,bounds=(0,1))

  def check_sat(self):
    s = z3.Solver()
    for clause in self.clauses:
      clause_vars = list(clause.keys())
      s.add(z3.Or(z3_literal(clause_vars[0],clause[clause_vars[0]]),z3_literal(clause_vars[1],clause[clause_vars[1]]),z3_literal(clause_vars[2],clause[clause_vars[2]])))
    return s.check()
