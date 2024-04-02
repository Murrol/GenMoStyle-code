import numpy as np


kinematic_chain = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[10,13,14,15,16],[10,17,18,19,20]]

parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 10, 13, 14, 15, 10, 17, 18, 19]

bfa_style_enumerator = {
    0:"Angry",
    1:"Depressed",
    2:"Drunk",
    3:"FemaleModel",
    4:"Happy",
    5:"Heavy",
    6:"Hurried",
    7:"Lazy",
    8:"Neutral",
    9:"Old",
    10:"Proud",
    11:"Robot",
    12:"Sneaky",
    13:"Soldier",
    14:"Strutting",
    15:"Zombie"
}
bfa_style_inv_enumerator = {value:key for key, value in bfa_style_enumerator.items()}
xia_style_enumerator = {
    0:"angry",
    1:"childlike",
    2:"depressed",
    3:"neutral",
    4:"old",
    5:"proud",
    6:"sexy",
    7:"strutting"
}

xia_action_enumerator={
    0:"fast_punching",
    1:"fast_walking",
    2:"jumping",
    3:"kicking",
    4:"punching",
    5:"running",
    6:"transitions",
    7:"normal_walking",
}

xia_style_inv_enumerator = {value:key for key, value in xia_style_enumerator.items()}
xia_action_inv_enumerator = {value:key for key, value in xia_action_enumerator.items()}