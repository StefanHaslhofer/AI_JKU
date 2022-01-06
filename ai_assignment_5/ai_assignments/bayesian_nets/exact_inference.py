import hedgehog as hh
import pandas as pd


def build_network():
    bn = hh.BayesNet(
        (['Relaxes', 'Whistles'], 'Sings'),
        ('Sings', 'Colorful'),
        (['Colorful', 'Vacation'], 'Happy'),
    )
    return bn


def initialize_probabilities(bn):
    bn.P['Relaxes'] = pd.Series({False: .6, True: .4})
    bn.P['Whistles'] = pd.Series({False: .7, True: .3})

    bn.P['Sings'] = pd.Series({
        (False, False, True): .4,  # P( Sings = True | Relaxes = False, Whistles = False )
        (False, True, True): .7,  # P( Sings = True | Relaxes = False, Whistles = True )
        (True, False, True): .1,  # P( Sings = True | Relaxes = True, Whistles = False )
        (True, True, True): .7,  # P( Sings = True | Relaxes = True, Whistles = True )

        (False, False, False): .6,
        (False, True, False): .3,
        (True, False, False): .9,
        (True, True, False): .3
    })

    bn.P['Colorful'] = pd.Series({
        (True, True): .6,  # P( Colorful = True | Sings = True )
        (False, True): .4,  # P( Colorful = True | Sings = False )

        (True, False): .01,
        (False, False): .99
    })

    bn.P['Vacation'] = pd.Series({False: .99, True: .01})

    bn.P['Happy'] = pd.Series({
        (False, False, True): .4,  # P( Happy = True | Vacation = False, Colorful = False )
        (False, True, True): .7,  # P( Happy = True | Vacation = False, Colorful = True )
        (True, False, True): .1,  # P( Happy = True | Vacation = True, Colorful = False )
        (True, True, True): .7,  # P( Happy = True | Vacation = True, Colorful = True )

        (False, False, False): .6,
        (False, True, False): .3,
        (True, False, False): .9,
        (True, True, False): .3
    })
    return bn


if __name__ == '__main__':
    bn = build_network()
    # optional: visualize network to check whether the structure is correct

    bn = initialize_probabilities(bn)
    bn.prepare()  # implemented in hedgehog, don't remove

    event = {'Whistles': True, 'Sings': False, 'Vacation': True, 'Happy': True}
    outcome = bn.query('Colorful', event=event)
    print('P(Colorful | Whistles, NOT Sings, Vacation, Happy) = ', '%.3f' % outcome[True], '\n')

    interm = bn.predict_proba({'Colorful': True, 'Whistles': True, 'Sings': True, 'Vacation': True, 'Happy': True})
    print('P(Colorful, Whistles, Sings, Vacation, Happy) =', '%.8f' % interm)
    interm = bn.predict_proba({'Colorful': False, 'Whistles': True, 'Sings': True, 'Vacation': True, 'Happy': True})
    print('P(NOT Colorful, Whistles, Sings, Vacation, Happy) =', '%.8f' % interm)

    event = {'Whistles': True, 'Sings': True, 'Vacation': True, 'Happy': True}
    alpha = 1 / bn.predict_proba(event)
    print('alpha =', '%.3f' % alpha)

    outcome = bn.query('Colorful', event=event)
    print('P(Colorful | Whistles, Sings, Vacation, Happy) = ', '%.3f' % outcome[True])
    print('P(NOT Colorful | Whistles, Sings, Vacation, Happy) = ', '%.3f' % outcome[False])

    dot = bn.graphviz()
    dot.attr(label=r'\n\n Exact Inference \n drawn by Stefan Haslhofer (K11908757 )')

    path = dot.render('exact_inference', directory='q2', format='png', cleanup=True)