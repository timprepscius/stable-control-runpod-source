INPUT_SCHEMA = {
    'seed': {
        'type': int,
        'required': False,
        'default': None
    },
    'prompt': {
    	'type': str,
        'required': True,
        'default': None
    },
    'negative_prompt': {
        'type': str,
        'required': True,
        'default': None
    },
    'width': {
        'type': int,
        'required': True,
        'default': None
    },
    'height': {
        'type': int,
        'required': True,
        'default': None
    },
    'guidance_scale': {
        'type': float,
        'required': False,
        'default': 7.5,
        'constraints': lambda guidance_scale: 0 < guidance_scale < 20
    },
    'width': {
        'type': int,
        'required': False,
        'default': 1024,
        'constraints': lambda width: width in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
    'height': {
        'type': int,
        'required': False,
        'default': 1024,
        'constraints': lambda height: height in [128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024]
    },
}
