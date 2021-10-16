# get content and style images
# CONTENT_IDX: 1: Neckarfront    2: sjtu entrance
#              3: Shenzhen scene 4: Anime scene     5: anime
# STYLE_IDX:   1: starry night   2: a colorful artwork with multiple geometries
#              3: abstract art   4: Mondrian art
CONTENT_IDX = 1
STYLE_IDX = 1

WEIGHT_CONTENT = 1
WEIGHTS_STYLE = [1e11, 1e11, 1e11, 1e10, 1e10, 1e9, 1e9]
# WEIGHTS_STYLE = [1e6] * len(layers)

max_iter = 500
show_iter = 50

content_layers = [37]
style_layers = [9, 12, 16, 22, 25, 29, 32]