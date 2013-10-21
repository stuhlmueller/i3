from i3 import marg


def test_marg():
  marginals_a = marg.Marginals(
    [0, 1, 2],
    [[.1, .9],
     [.2, .8],
     [.3, .7]])
  marginals_b = marg.Marginals(
    [0, 1, 2],
    [[.11, .89],
     [.19, .81],
     [.31, .69]])
  assert marginals_a - marginals_b < .02
  assert marginals_a - marginals_b <= .02
  assert marginals_a - marginals_b > .001
  assert marginals_a - marginals_b >= .001
  assert not marginals_a - marginals_b < .01
  assert marginals_a > None
  assert not marginals_a < None
