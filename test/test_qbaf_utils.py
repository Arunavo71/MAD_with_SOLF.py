from mad.qbaf_utils import *

def test_get_untargeted_args():
    args = ['a', 'b', 'c', 'd', 'e']
    initial_strengths = [0, 0, 0, 0, 0]
    atts = [('b', 'a'), ('e', 'c')]
    supps = [('c', 'a'), ('d', 'b')]
    qbaf = QBAFramework(args, initial_strengths, atts, supps, semantics="DFQuAD_model")
    untargeted_args = get_untargeted_args(qbaf)
    assert 'd' in untargeted_args
    assert 'e' in untargeted_args
    assert 'a' not in untargeted_args
    assert 'b' not in untargeted_args
    assert 'c' not in untargeted_args