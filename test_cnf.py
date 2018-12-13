import unittest
import cnf
import z3

class TestCnf(unittest.TestCase):
  def test_is_ktuple_even(self):
    tuple1 = [{1:True, 2:False, 4:True}, {3:True,2:False,1:True},{3:False,4:False,5:True}]
    self.assertFalse(cnf.is_ktuple_even(tuple1))
    tuple2 = [{1:True,2:False,3:True},{3:False,2:True,1:False}]
    self.assertTrue(cnf.is_ktuple_even(tuple2))

  def test_is_ktuple_inconsistent(self):
    tuple1 = [{1:True, 2:False, 4:True}, {3:True,2:False,1:False},{3:False,4:False,5:True}]
    self.assertTrue(cnf.is_ktuple_inconsistent(tuple1))
    tuple2 = [{1:True,2:False,3:True},{3:False,2:False,1:False}]
    self.assertFalse(cnf.is_ktuple_inconsistent(tuple2))

  def test_choose_random_polarity(self):
    p = cnf.choose_random_polarity()
    self.assertTrue(p == True or p == False)

  def test_get_random_phi(self):
    cnf3 = cnf.ThreeCNF(10,30)
    cnf3.get_random_phi()
    self.assertEqual(len(cnf3.clauses),cnf3.m)

  def test_find_imbalance(self):
    cnf3 = cnf.ThreeCNF(5,10)
    cnf3.clauses = [{3: True, 2: True, 0: True}, {2: True, 0: False, 1: True}, {3: False, 2: False, 4: False}, {1: False, 4: True, 0: True}, {0: False, 1: False, 2: True}, {4: True, 2: True, 0: False}, {3: False, 0: True, 1: True}, {1: True, 2: True, 3: False}, {2: False, 3: False, 4: False}, {4: True, 3: False, 2: False}]
    self.assertEqual(cnf3.find_imbalance(), 8)

  def test_calculate_m_phi(self):
    cnf3 = cnf.ThreeCNF(5,3)
    cnf3.clauses = [{1:True,2:False,0:False},{2:True,0:True,4:False},{3:True,0:False,2:False}]
    m_phi = cnf3.calculate_m_phi()
    for i in range(cnf3.n):
      self.assertEqual(m_phi[i][i],0)
    self.assertEqual(m_phi[0][2],-1.5)
    self.assertEqual(m_phi[2][0],-1.5)
    self.assertEqual(m_phi[2][4],0.5)
    self.assertEqual(m_phi[1][4],0)

  def test_check_sat(self):
    cnf3 = cnf.ThreeCNF(3,2)
    cnf3.clauses = [{0:False,1:False,2:False},{0:True,1:True,2:True}]
    self.assertEqual(cnf3.check_sat(),z3.sat)

if __name__ == '__main__':
  unittest.main()