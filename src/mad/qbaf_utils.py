from qbaf import QBAFramework

def get_untargeted_args(qbaf: QBAFramework) -> list[str]:
    """Returns all arguments in a given QBAF that have neither attackers nor supporters.

    Args:
        qbaf (QBAFramework): QBAF whose untargeted arguments should be returned.

    Returns:
        list[str]: list of untargeted argument identifiers.
    """
    return [arg for arg in qbaf.arguments if len(qbaf.attackersOf(arg)) == 0 and len(qbaf.supportersOf(arg)) == 0]