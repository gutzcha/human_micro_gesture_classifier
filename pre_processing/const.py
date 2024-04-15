'''
 head:		hnod, hshk, hmov
    lobo: 		lgcr, lgucr, fmov
    hands/arms:	hrep, fidg, htfa, hthd
    upbo:		flean, blean, shrai, shrg
    fubo:		readj

    shrg is combimend with shrai! 
'''
TIER2LABEL = dict(
    head = ['hnod', 'hshk', 'hmov'],
    hands_arms = ['hrep', 'fidg', 'htfa', 'hthd'],
    upbo = ['flean', 'blean', 'shrai'],
    fubo = ['readj']
)

LABEL2TIER = {label: tier for tier, labels in TIER2LABEL.items() for label in labels}

FEATURE_NAMES = ['hnod', 'hshk', 'hmov', 'lgcr', 'lgucr', 'fmov', 'fidg', 'htfa',
       'hthd', 'flean', 'blean', 'shrai', 'readj']