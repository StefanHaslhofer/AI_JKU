import hedgehog as hh

# adventure is seperate because it doesn´t influence the study decision
bn = hh.BayesNet(
    ('Sunny', 'Party'),
    ('Weekend', 'Party'),
    (['Weekend', 'Party', 'Likes subject'], 'Study'),
    ('Adventure')
)

dot = bn.graphviz()
dot.attr(label=r'\n\n Bayesian Network \n drawn by Stefan Haslhofer (K11908757 )')

path = dot.render('bayesian_net', directory='q1', format='png', cleanup=True)
