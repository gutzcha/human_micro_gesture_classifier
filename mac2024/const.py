ID2LABELS = {
    0: "shaking body",
    1: "sitting straightly",
    2: "shrugging",
    3: "turning around",
    4: "rising up",
    5: "bowing head",
    6: "head up",
    7: "tilting head",
    8: "turning head",
    9: "nodding",
    10: "shaking head",
    11: "scratching arms",
    12: "playing objects",
    13: "putting hands together",
    14: "rubbing hands",
    15: "pointing oneself",
    16: "clenching fist",
    17: "stretching arms",
    18: "retracting arms",
    19: "waving",
    20: "spreading hands",
    21: "hands touching fingers",
    22: "other finger movements",
    23: "illustrative gestures",
    24: "shaking legs",
    25: "curling legs",
    26: "spread legs",
    27: "closing legs",
    28: "crossing legs",
    29: "stretching feet",
    30: "retracting feet",
    31: "tiptoe",
    32: "scratching or touching neck",
    33: "scratching or touching chest",
    34: "scratching or touching back",
    35: "scratching or touching shoulder",
    36: "arms akimbo",
    37: "crossing arms",
    38: "playing or tidying hair",
    39: "scratching or touching hindbrain",
    40: "scratching or touching forehead",
    41: "scratching or touching face",
    42: "rubbing eyes",
    43: "touching nose",
    44: "touching ears",
    45: "covering face",
    46: "covering mouth",
    47: "pushing glasses",
    48: "patting legs",
    49: "touching legs",
    50: "scratching legs",
    51: "scratching feet"
}

COARSE_LABEL = {
    0: 'body',
    1: 'head',
    2: 'upper limb',
    3: 'lower limb',
    4: 'body-hand',
    5: 'head-hand',
    6: 'leg-hand'
}


def fine2coarse(num):
    if 0 <= num <= 4:
        return 0
    elif 5 <= num <= 10:
        return 1
    elif 11 <= num <= 23:
        return 2
    elif 24 <= num <= 31:
        return 3
    elif 32 <= num <= 37:
        return 4
    elif 38 <= num <= 47:
        return 5
    elif 48 <= num <= 51:
        return 6


COARSE2FINE = {
    0: range(0, 5),
    1: range(5, 11),
    2: range(11, 24),
    3: range(24, 32),
    4: range(32, 38),
    5: range(38, 46),
    6: range(48, 52)
}
